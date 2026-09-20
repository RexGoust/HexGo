use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use burn::Tensor;
use clap::Args;
use hex_go::ai::backend::{CpuBackend, CudaBackend, cpu_device, cuda_device};
use hex_go::ai::burn_neural_network::BurnNeuralNetwork;
use hex_go::ai::dummy_network::DummyNetwork;
use hex_go::ai::encoder::{adjacency_tensor, encode_game_gnn, encode_game_mlp};
use hex_go::ai::model::{HexGoModel, MlpModelConfig, ModelConfig, gnn::GnnModelConfig};
use hex_go::ai::neural_mcts::{NeuralConfig, NeuralMcts, StepState};
use hex_go::ai::neural_network::NeuralNetwork;
use hex_go::ai::search::Search;
use hex_go::ai::search_result::SearchResult;
use hex_go::board_layout::BoardDefinition;
use hex_go::game::Game;
use rayon::prelude::*;

use crate::active_game::create_game;
use crate::device::DeviceKind;
use crate::game_slot::GameSlot;
use crate::model_type::ModelType;
use crate::self_play::generate_samples_local;
use crate::train_network::forward_batch;

#[derive(Args, Clone, Debug)]
pub struct BenchArgs {
    /// Model architecture type to benchmark (mlp / gnn)
    #[arg(long, value_enum, default_value_t = ModelType::Gnn)]
    pub model_type: ModelType,

    /// Execution device to benchmark (cpu / cuda)
    #[arg(long, value_enum, default_value_t = DeviceKind::Cpu)]
    pub infer_device: DeviceKind,

    /// Batch size for batched inference benchmark (Only relevant for CUDA engine)
    #[arg(long, default_value_t = 256)]
    pub infer_size: usize,

    /// Number of games for pipeline breakdown benchmark
    #[arg(short, long, default_value_t = 10)]
    pub games: usize,

    /// Number of MCTS iterations per move for pipeline breakdown benchmark
    #[arg(short, long, default_value_t = 100)]
    pub iterations: usize,

    /// Run benchmarks for both CPU and CUDA side-by-side for comparison
    #[arg(long, default_value_t = false)]
    pub compare: bool,
}

/// 1. Benchmark pure board game rules logic without any neural network.
pub fn bench_game_logic() {
    println!("\n================================================================================");
    println!("  [Benchmark 1/3] Raw Game Board Engine Performance (No Neural Network)");
    println!("================================================================================");

    let game = create_game();

    // Measure Game::clone() performance
    const CLONE_ITERS: usize = 500_000;
    let t0 = Instant::now();
    let mut sink = 0;
    for _ in 0..CLONE_ITERS {
        let g = game.clone();
        sink += g.current_player() as usize;
    }
    let clone_dur = t0.elapsed();
    let clone_qps = CLONE_ITERS as f64 / clone_dur.as_secs_f64();
    let ns_per_clone = (clone_dur.as_nanos() as f64) / (CLONE_ITERS as f64);
    println!(
        "* Game::clone()           : {:>8} iters in {:>8.2?} | {:>10.2} ops/sec (sink: {}) | {:>8.1} ns/op",
        CLONE_ITERS, clone_dur, clone_qps, sink, ns_per_clone
    );

    // Measure play_move() performance with random rollouts
    const ROLLOUT_GAMES: usize = 1_000;
    let mut total_moves = 0;
    let t1 = Instant::now();
    for _ in 0..ROLLOUT_GAMES {
        let mut g = create_game();
        while g.result().is_none() && total_moves < 500_000 {
            let moves = g.legal_moves();
            if moves.is_empty() {
                let _ = g.pass_turn();
            } else {
                let m = moves[total_moves % moves.len()];
                let _ = g.play_move(m);
            }
            total_moves += 1;
        }
    }
    let move_dur = t1.elapsed();
    let move_qps = total_moves as f64 / move_dur.as_secs_f64();
    let ns_per_move = (move_dur.as_nanos() as f64) / (total_moves as f64);
    println!(
        "* Game::play_move() (liberty): {:>8} moves in {:>8.2?} | {:>10.2} moves/sec       | {:>8.1} ns/op",
        total_moves, move_dur, move_qps, ns_per_move
    );
}

/// 2A. Benchmark independent single-sample CPU inference (the actual path taken by CPU self-play).
pub fn bench_cpu_independent_inference(model_type: ModelType) {
    println!("\n================================================================================");
    println!("  [Benchmark 2/3] Raw CPU Independent Inference Performance (No Batching)");
    println!("================================================================================");
    println!("* Note: CPU self-play runs independent single-sample evaluations (batch size = 1)");
    println!("  across parallel Rayon threads. No batching or PCIe transfer is involved.\n");

    let device = cpu_device();
    let config = match model_type {
        ModelType::Mlp => ModelConfig::Mlp(MlpModelConfig::default()),
        ModelType::Gnn => ModelConfig::Gnn(GnnModelConfig::default()),
    };
    let model = HexGoModel::<CpuBackend>::new(config, &device);
    let network = Arc::new(BurnNeuralNetwork::from_model(&model));
    let test_game = create_game();

    // Warmup evaluations
    for _ in 0..20 {
        let _ = network.evaluate(&test_game, test_game.current_player());
    }

    // 1. Single-threaded sequential evaluation (1 core)
    const SINGLE_ITERS: usize = 500;
    let t0 = Instant::now();
    for _ in 0..SINGLE_ITERS {
        let _ = network.evaluate(&test_game, test_game.current_player());
    }
    let single_elapsed = t0.elapsed();
    let avg_single_us = (single_elapsed.as_micros() as f64) / (SINGLE_ITERS as f64);
    let single_qps = (SINGLE_ITERS as f64) / single_elapsed.as_secs_f64();

    println!(
        "* Single-Thread Latency (1 core, batch=1)   : {:>8.2} us/eval   | {:>10.0} evals/sec",
        avg_single_us, single_qps
    );

    // 2. Multi-threaded parallel independent evaluation (Rayon)
    let num_threads = rayon::current_num_threads();
    let multi_iters = num_threads * 200;

    // Warmup Rayon threads
    (0..num_threads * 5).into_par_iter().for_each(|_| {
        let _ = network.evaluate(&test_game, test_game.current_player());
    });

    let t1 = Instant::now();
    (0..multi_iters).into_par_iter().for_each(|_| {
        let _ = network.evaluate(&test_game, test_game.current_player());
    });
    let multi_elapsed = t1.elapsed();
    let multi_qps = (multi_iters as f64) / multi_elapsed.as_secs_f64();
    let avg_multi_us =
        (multi_elapsed.as_micros() as f64 * num_threads as f64) / (multi_iters as f64);
    let speedup = multi_qps / single_qps;
    let efficiency = (speedup / (num_threads as f64)) * 100.0;

    println!(
        "* Multi-Thread Throughput ({} Rayon threads) : {:>8.2} us/sample | {:>10.0} evals/sec",
        num_threads, avg_multi_us, multi_qps
    );
    println!(
        "* Multi-Thread Scaling Speedup              : {:>8.2}x / {} cores ({:>5.1}% scaling efficiency)",
        speedup, num_threads, efficiency
    );
    println!("--------------------------------------------------------------------------------");
}

/// 2B. Benchmark CUDA batched forward passes across varying batch sizes.
pub fn bench_cuda_batch_scaling(model_type: ModelType) {
    println!("\n================================================================================");
    println!(
        "  [Benchmark 2/3] Raw CUDA Forward Inference Scaling ({:?})",
        model_type
    );
    println!("================================================================================");
    println!(
        "{:<12} {:<18} {:<18} {:<18}",
        "Batch Size", "Batch Time (us)", "Per-Sample (us)", "Throughput (samples/s)"
    );
    println!("--------------------------------------------------------------------------------");

    let device = cuda_device();
    let config = match model_type {
        ModelType::Mlp => ModelConfig::Mlp(MlpModelConfig::default()),
        ModelType::Gnn => ModelConfig::Gnn(GnnModelConfig::default()),
    };
    let model = HexGoModel::<CudaBackend>::new(config, &device);
    let adj = match &model {
        HexGoModel::Gnn(_) => {
            adjacency_tensor::<CudaBackend>(BoardDefinition::compact().graph(), &device)
        }
        _ => Tensor::zeros([1, 1], &device),
    };

    let test_game = create_game();
    let dummy_input = match model_type {
        ModelType::Mlp => encode_game_mlp(&test_game, test_game.current_player()),
        ModelType::Gnn => encode_game_gnn(&test_game, test_game.current_player()),
    };

    let batch_sizes = [1, 16, 32, 64, 128, 256, 512];

    for &bs in &batch_sizes {
        let inputs: Vec<&Vec<f32>> = (0..bs).map(|_| &dummy_input).collect();

        // Warmup runs
        for _ in 0..5 {
            let _ = forward_batch(&model, &device, &adj, &inputs);
        }

        // Timed benchmark runs
        const REPEATS: u32 = 50;
        let t0 = Instant::now();
        for _ in 0..REPEATS {
            let _ = forward_batch(&model, &device, &adj, &inputs);
        }
        let elapsed = t0.elapsed();
        let avg_batch_us = (elapsed.as_micros() as f64) / (REPEATS as f64);
        let per_sample_us = avg_batch_us / (bs as f64);
        let throughput = (bs as f64 * REPEATS as f64) / elapsed.as_secs_f64();

        println!(
            "{:<12} {:<18.2} {:<18.2} {:<18.0}",
            bs, avg_batch_us, per_sample_us, throughput
        );
    }
    println!("--------------------------------------------------------------------------------");
}

#[derive(Default)]
pub struct CpuPipelineStats {
    pub select_ns: AtomicU64,
    pub eval_ns: AtomicU64,
    pub update_ns: AtomicU64,
    pub total_thread_work_ns: AtomicU64,
    pub eval_count: AtomicU64,
    pub search_calls: AtomicU64,
}

struct ProfilingSearch {
    mcts: NeuralMcts<DummyNetwork>,
    network: Arc<BurnNeuralNetwork>,
    stats: Arc<CpuPipelineStats>,
    start_time: Instant,
}

impl Search for ProfilingSearch {
    fn search(&mut self, game: &Game, iterations: usize) -> Option<SearchResult> {
        self.stats.search_calls.fetch_add(1, Ordering::Relaxed);
        if !self.mcts.start_search(game) {
            return None;
        }

        for _ in 0..iterations {
            let t_sel = Instant::now();
            let step = self.mcts.step_select(game);
            let sel_dur = t_sel.elapsed().as_nanos() as u64;
            self.stats.select_ns.fetch_add(sel_dur, Ordering::Relaxed);

            match step {
                StepState::Terminal => continue,
                StepState::NeedsEvaluation { node, leaf_game } => {
                    let player = leaf_game.current_player();

                    let t_eval = Instant::now();
                    let evaluation = self.network.evaluate(&leaf_game, player);
                    let eval_dur = t_eval.elapsed().as_nanos() as u64;
                    self.stats.eval_ns.fetch_add(eval_dur, Ordering::Relaxed);
                    self.stats.eval_count.fetch_add(1, Ordering::Relaxed);

                    let t_up = Instant::now();
                    self.mcts
                        .step_update(node, &leaf_game, &evaluation.policy, evaluation.value);
                    let up_dur = t_up.elapsed().as_nanos() as u64;
                    self.stats.update_ns.fetch_add(up_dur, Ordering::Relaxed);
                }
                StepState::Initial => continue,
            }
        }

        self.mcts.finish_search()
    }
}

impl Drop for ProfilingSearch {
    fn drop(&mut self) {
        let game_dur = self.start_time.elapsed().as_nanos() as u64;
        self.stats
            .total_thread_work_ns
            .fetch_add(game_dur, Ordering::Relaxed);
    }
}

/// 3A. Benchmark native CPU self-play pipeline with microsecond stage breakdown.
pub fn bench_cpu_pipeline_breakdown(args: &BenchArgs) {
    println!("\n================================================================================");
    println!("  [Benchmark 3/3] Native CPU Self-Play Pipeline Breakdown (Rayon Parallel)");
    println!("================================================================================");

    if args.games == 0 {
        println!("Skipping pipeline benchmark (games = 0).");
        return;
    }

    let device = cpu_device();
    let config = match args.model_type {
        ModelType::Mlp => ModelConfig::Mlp(MlpModelConfig::default()),
        ModelType::Gnn => ModelConfig::Gnn(GnnModelConfig::default()),
    };
    let model = HexGoModel::<CpuBackend>::new(config, &device);
    let network = Arc::new(BurnNeuralNetwork::from_model(&model));
    let stats = Arc::new(CpuPipelineStats::default());

    let num_threads = rayon::current_num_threads();
    println!(
        "Running {} games, {} iterations/move across {} Rayon worker threads...",
        args.games, args.iterations, num_threads
    );

    let overall_start = Instant::now();
    let samples = {
        let network_clone = Arc::clone(&network);
        let stats_clone = Arc::clone(&stats);
        generate_samples_local(args.model_type, args.games, args.iterations, move || {
            ProfilingSearch {
                mcts: NeuralMcts::new(DummyNetwork, NeuralConfig { add_noise: true }),
                network: Arc::clone(&network_clone),
                stats: Arc::clone(&stats_clone),
                start_time: Instant::now(),
            }
        })
    };
    let overall_dur = overall_start.elapsed();

    let total_thread_work_ns = stats.total_thread_work_ns.load(Ordering::Relaxed) as u128;
    let eval_ns = stats.eval_ns.load(Ordering::Relaxed) as u128;
    let select_ns = stats.select_ns.load(Ordering::Relaxed) as u128;
    let update_ns = stats.update_ns.load(Ordering::Relaxed) as u128;
    let total_evals = stats.eval_count.load(Ordering::Relaxed);
    let search_calls = stats.search_calls.load(Ordering::Relaxed);

    let mcts_work_ns = eval_ns + select_ns + update_ns;
    let root_work_ns = total_thread_work_ns.saturating_sub(mcts_work_ns);
    let total_work_ns = total_thread_work_ns.max(mcts_work_ns);

    let wall_secs = overall_dur.as_secs_f64();
    let samples_per_sec = if wall_secs > 0.0 {
        (samples.len() as f64) / wall_secs
    } else {
        0.0
    };
    let cpu_work_secs = (total_work_ns as f64) / 1_000_000_000.0;
    let effective_parallelism = if wall_secs > 0.0 {
        cpu_work_secs / wall_secs
    } else {
        0.0
    };

    println!(
        "Benchmark Complete! Overall Wall-Clock Time: {:.2?}",
        overall_dur
    );
    println!(
        "Completed Games: {} | Generated Samples: {} | Move Decisions: {}",
        args.games,
        samples.len(),
        search_calls
    );
    println!(
        "Total Evaluated Positions: {} | Wall-Clock Throughput: {:.2} samples/sec",
        total_evals, samples_per_sec
    );
    println!(
        "Total CPU Thread Work: {:.2}s across {} threads (effective parallelism: {:.1}x)",
        cpu_work_secs, num_threads, effective_parallelism
    );

    println!(
        "\n{:<38} {:<18} {:<18} {:<12}",
        "Stage", "Total CPU Time (ms)", "Avg / Call (us)", "Percentage"
    );
    println!("--------------------------------------------------------------------------------");

    let print_row = |name: &str, ns: u128, count: u64| {
        let ms = (ns as f64) / 1_000_000.0;
        let avg_us = if count > 0 {
            (ns as f64) / (count as f64 * 1000.0)
        } else {
            0.0
        };
        let pct = if total_work_ns > 0 {
            (ns as f64 / total_work_ns as f64) * 100.0
        } else {
            0.0
        };
        println!("{:<38} {:<18.2} {:<18.2} {:<11.1}%", name, ms, avg_us, pct);
    };

    print_row("1. CPU NN Evaluate (Forward+Encode)", eval_ns, total_evals);
    print_row("2. MCTS Selection (Tree+Clone)", select_ns, total_evals);
    print_row("3. MCTS Update (Expand+Backprop)", update_ns, total_evals);
    print_row(
        "4. Root Move Step & Sample Buffer",
        root_work_ns,
        search_calls,
    );
    println!("================================================================================");

    let eval_pct = if total_work_ns > 0 {
        (eval_ns as f64 / total_work_ns as f64) * 100.0
    } else {
        0.0
    };
    let avg_eval_us = if total_evals > 0 {
        (eval_ns as f64) / (total_evals as f64 * 1000.0)
    } else {
        0.0
    };

    println!("* Diagnostic Summary:");
    println!(
        "  - Neural network inference consumes {:.1}% of total CPU self-play compute.",
        eval_pct
    );
    println!(
        "  - Each evaluation takes ~{:.1} us. With {} iterations/move and ~{} moves/game,",
        avg_eval_us,
        args.iterations,
        if args.games > 0 {
            search_calls / args.games as u64
        } else {
            0
        }
    );
    println!(
        "    each game requires ~{} NN evaluations.",
        if args.games > 0 {
            total_evals / args.games as u64
        } else {
            0
        }
    );
    println!("================================================================================\n");
}

/// 3B. Benchmark CUDA double-buffered batched pipeline with stage breakdown.
pub fn bench_cuda_pipeline_breakdown(args: &BenchArgs) {
    println!("\n================================================================================");
    println!("  [Benchmark 3/3] CUDA Double-Buffered Batched Pipeline Stage Breakdown");
    println!("================================================================================");

    if args.games == 0 {
        println!("Skipping pipeline benchmark (games = 0).");
        return;
    }

    let device = cuda_device();
    let config = match args.model_type {
        ModelType::Mlp => ModelConfig::Mlp(MlpModelConfig::default()),
        ModelType::Gnn => ModelConfig::Gnn(GnnModelConfig::default()),
    };
    let model = HexGoModel::<CudaBackend>::new(config, &device);
    let adj = match &model {
        HexGoModel::Gnn(_) => {
            adjacency_tensor::<CudaBackend>(BoardDefinition::compact().graph(), &device)
        }
        _ => Tensor::zeros([1, 1], &device),
    };

    let initial_games = args.infer_size.min(args.games);
    let count_0 = initial_games.div_ceil(2);
    let count_1 = initial_games - count_0;

    let mut slots = [GameSlot::new(count_0), GameSlot::new(count_1)];
    let mut started = slots[0].games.len() + slots[1].games.len();
    let mut completed = 0;
    let mut samples = Vec::new();

    // Stage timer accumulators in nanoseconds
    let mut select_ns = 0u128;
    let mut forward_ns = 0u128;
    let mut update_ns = 0u128;
    let mut queue_wait_ns = 0u128;
    let mut commit_ns = 0u128;
    let mut forward_calls = 0u64;
    let mut total_evaluated_samples = 0u64;

    let overall_start = Instant::now();

    std::thread::scope(|s| {
        let (req_tx, req_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<Vec<f32>>)>(2);
        let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel(2);

        // Dedicated GPU Worker thread tracking raw forward execution time
        s.spawn(move || {
            while let Ok((id, inputs)) = req_rx.recv() {
                let num_samples = inputs.len();
                let refs: Vec<&Vec<f32>> = inputs.iter().collect();
                let t_start = Instant::now();
                let res = forward_batch(&model, &device, &adj, &refs);
                let duration_ns = t_start.elapsed().as_nanos();

                if resp_tx.send((id, res, duration_ns, num_samples)).is_err() {
                    break;
                }
            }
        });

        // Pipeline warmup
        let mut in_flight = 0;
        for (id, slot) in slots.iter_mut().enumerate() {
            let t_sel = Instant::now();
            let inputs = slot.select(args.model_type);
            select_ns += t_sel.elapsed().as_nanos();

            if !inputs.is_empty() {
                req_tx.send((id, inputs)).unwrap();
                in_flight += 1;
            }
        }

        // Pipelined processing loop
        while in_flight > 0 {
            let t_wait = Instant::now();
            let (id, (policies, values), f_ns, num_samples) = resp_rx.recv().unwrap();
            queue_wait_ns += t_wait.elapsed().as_nanos();
            forward_ns += f_ns;
            forward_calls += 1;
            total_evaluated_samples += num_samples as u64;
            in_flight -= 1;

            let slot = &mut slots[id];

            // 1. MCTS tree backpropagation
            let t_up = Instant::now();
            slot.update(&policies, &values);
            update_ns += t_up.elapsed().as_nanos();

            slot.step += 1;

            // 2. Action commitment if iterations reached
            if slot.step >= args.iterations {
                let t_com = Instant::now();
                slot.commit_actions(
                    args.model_type,
                    &mut samples,
                    &mut completed,
                    &mut started,
                    args.games,
                );
                commit_ns += t_com.elapsed().as_nanos();
            }

            // 3. Next MCTS selection step
            if !slot.games.is_empty() && completed < args.games {
                loop {
                    let t_sel = Instant::now();
                    let next_inputs = slot.select(args.model_type);
                    select_ns += t_sel.elapsed().as_nanos();

                    if !next_inputs.is_empty() {
                        req_tx.send((id, next_inputs)).unwrap();
                        in_flight += 1;
                        break;
                    }

                    slot.step += 1;
                    if slot.step >= args.iterations {
                        let t_com = Instant::now();
                        slot.commit_actions(
                            args.model_type,
                            &mut samples,
                            &mut completed,
                            &mut started,
                            args.games,
                        );
                        commit_ns += t_com.elapsed().as_nanos();
                    }

                    if slot.games.is_empty() || completed >= args.games {
                        break;
                    }
                }
            }
        }

        drop(req_tx);
    });

    let overall_dur = overall_start.elapsed();
    let total_ns = overall_dur.as_nanos() as f64;
    let avg_batch_size = if forward_calls > 0 {
        total_evaluated_samples as f64 / forward_calls as f64
    } else {
        0.0
    };

    println!("Benchmark Complete! Overall Time: {:.2?}", overall_dur);
    println!(
        "Completed Games: {} | Generated Samples: {} | Batch Forward Calls: {}",
        completed,
        samples.len(),
        forward_calls
    );
    println!(
        "Total Evaluated Positions: {} | Average Batch Size per Forward Call: {:.2}",
        total_evaluated_samples, avg_batch_size
    );

    println!(
        "\n{:<35} {:<16} {:<18} {:<12}",
        "Stage", "Total Time (ms)", "Avg / Call (us)", "Percentage"
    );
    println!("--------------------------------------------------------------------------------");

    let print_row = |name: &str, ns: u128| {
        let ms = (ns as f64) / 1_000_000.0;
        let avg_us = if forward_calls > 0 {
            (ns as f64) / (forward_calls as f64 * 1000.0)
        } else {
            0.0
        };
        let pct = if total_ns > 0.0 {
            (ns as f64 / total_ns) * 100.0
        } else {
            0.0
        };
        println!("{:<35} {:<16.2} {:<18.2} {:<11.1}%", name, ms, avg_us, pct);
    };

    print_row("1. CPU Select (Tree Search+Clone)", select_ns);
    print_row("2. GPU Forward (Inference+PCIe)", forward_ns);
    print_row("3. CPU Update (Backpropagation)", update_ns);
    print_row("4. Channel Wait (Queue Sync)", queue_wait_ns);
    print_row("5. Commit Action (Move+Finish)", commit_ns);
    println!("================================================================================\n");
}

/// Entry point to execute the appropriate benchmark suite based on CLI options.
pub fn run_benchmark(args: BenchArgs) {
    println!("Starting HexGo Benchmark Suite with config: {:?}", args);

    // 1. Benchmark pure board game rules (always runs)
    bench_game_logic();

    if args.compare {
        // Run both CPU and CUDA benchmarks side-by-side
        bench_cpu_independent_inference(args.model_type);
        bench_cuda_batch_scaling(args.model_type);
        bench_cpu_pipeline_breakdown(&args);
        bench_cuda_pipeline_breakdown(&args);
        return;
    }

    // 2 & 3. Device-specific benchmark routing
    match args.infer_device {
        DeviceKind::Cpu => {
            // Benchmark native CPU independent evaluations (batch size = 1)
            bench_cpu_independent_inference(args.model_type);
            // Benchmark real multi-threaded CPU self-play pipeline with breakdown
            bench_cpu_pipeline_breakdown(&args);
        }
        DeviceKind::Cuda => {
            // Benchmark CUDA forward scaling across batch sizes
            bench_cuda_batch_scaling(args.model_type);
            // Benchmark CUDA double-buffered batched pipeline breakdown
            bench_cuda_pipeline_breakdown(&args);
        }
    }
}
