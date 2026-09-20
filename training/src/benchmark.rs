use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use burn::Tensor;
use clap::Args;
use hex_go::ai::backend::{CpuBackend, CudaBackend, cpu_device, cuda_device, switch_model_backend};
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
use crate::train_network::{GPU_MIN_BATCH, forward_batch, forward_batch_cpu_parallel};

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

    /// Compare pipeline performance with vs without CPU dynamic fallback
    #[arg(long, default_value_t = false)]
    pub compare_fallback: bool,
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

/// 2B. Benchmark CUDA vs CPU parallel forward inference across varying batch sizes.
pub fn bench_cuda_batch_scaling(model_type: ModelType) {
    println!("\n================================================================================");
    println!(
        "  [Benchmark 2/3] Raw CUDA vs CPU Parallel Forward Inference Scaling ({:?})",
        model_type
    );
    println!("================================================================================");
    println!(
        "{:<10} {:<16} {:<16} {:<18} {:<18} {:<14}",
        "Batch Size",
        "GPU Latency (us)",
        "CPU Latency (us)",
        "Speedup / Winner",
        "GPU Tput (sps)",
        "CPU Tput (sps)"
    );
    println!("--------------------------------------------------------------------------------");

    let cuda_dev = cuda_device();
    let cpu_dev = cpu_device();
    let config = match model_type {
        ModelType::Mlp => ModelConfig::Mlp(MlpModelConfig::default()),
        ModelType::Gnn => ModelConfig::Gnn(GnnModelConfig::default()),
    };
    let cuda_model = HexGoModel::<CudaBackend>::new(config, &cuda_dev);
    let cpu_model = switch_model_backend::<CudaBackend, CpuBackend>(cuda_model.clone(), &cpu_dev);

    let cuda_adj = match &cuda_model {
        HexGoModel::Gnn(_) => {
            adjacency_tensor::<CudaBackend>(BoardDefinition::compact().graph(), &cuda_dev)
        }
        _ => Tensor::zeros([1, 1], &cuda_dev),
    };
    let cpu_adj = match &cpu_model {
        HexGoModel::Gnn(_) => {
            let board = BoardDefinition::compact().graph().clone();
            adjacency_tensor::<CpuBackend>(&board, &cpu_dev)
        }
        _ => Tensor::zeros([1, 1], &cpu_dev),
    };

    let test_game = create_game();
    let dummy_input = match model_type {
        ModelType::Mlp => encode_game_mlp(&test_game, test_game.current_player()),
        ModelType::Gnn => encode_game_gnn(&test_game, test_game.current_player()),
    };

    let batch_sizes = [1, 4, 8, 16, 32, 48, 64, 96, 128, 256];

    for &bs in &batch_sizes {
        let inputs_owned: Vec<Vec<f32>> = (0..bs).map(|_| dummy_input.clone()).collect();
        let inputs_refs: Vec<&Vec<f32>> = inputs_owned.iter().collect();

        // Warmup GPU
        for _ in 0..5 {
            let _ = forward_batch(&cuda_model, &cuda_dev, &cuda_adj, &inputs_refs);
        }
        // Warmup CPU
        for _ in 0..3 {
            let _ = forward_batch_cpu_parallel(&cpu_model, &cpu_dev, &cpu_adj, &inputs_owned);
        }

        // Timed GPU benchmark
        const REPEATS: u32 = 30;
        let t0 = Instant::now();
        for _ in 0..REPEATS {
            let _ = forward_batch(&cuda_model, &cuda_dev, &cuda_adj, &inputs_refs);
        }
        let gpu_dur = t0.elapsed();
        let gpu_avg_us = (gpu_dur.as_micros() as f64) / (REPEATS as f64);
        let gpu_tput = (bs as f64 * REPEATS as f64) / gpu_dur.as_secs_f64();

        // Timed CPU Parallel benchmark
        let t1 = Instant::now();
        for _ in 0..REPEATS {
            let _ = forward_batch_cpu_parallel(&cpu_model, &cpu_dev, &cpu_adj, &inputs_owned);
        }
        let cpu_dur = t1.elapsed();
        let cpu_avg_us = (cpu_dur.as_micros() as f64) / (REPEATS as f64);
        let cpu_tput = (bs as f64 * REPEATS as f64) / cpu_dur.as_secs_f64();

        let winner = if cpu_avg_us < gpu_avg_us {
            let ratio = gpu_avg_us / cpu_avg_us;
            format!("{:.2}x CPU Win", ratio)
        } else {
            let ratio = cpu_avg_us / gpu_avg_us;
            format!("{:.2}x GPU Win", ratio)
        };

        println!(
            "{:<10} {:<16.2} {:<16.2} {:<18} {:<18.0} {:<14.0}",
            bs, gpu_avg_us, cpu_avg_us, winner, gpu_tput, cpu_tput
        );
    }
    println!("--------------------------------------------------------------------------------");
    println!(
        "* Analysis: CPU parallel dominates for batch < 64 (bypasses GPU's ~740us latency floor),"
    );
    println!("  while GPU throughput scales up dramatically for batch >= 64.");
    println!("================================================================================\n");
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

#[derive(Default)]
pub struct PipelineRunStats {
    pub mode_name: &'static str,
    pub overall_dur: std::time::Duration,
    pub completed: usize,
    pub samples_len: usize,
    pub forward_calls: u64,
    pub gpu_forward_calls: u64,
    pub cpu_forward_calls: u64,
    pub total_evaluated_samples: u64,
    pub select_ns: u128,
    pub forward_ns: u128,
    pub update_ns: u128,
    pub queue_wait_ns: u128,
    pub commit_ns: u128,
}

fn run_cuda_pipeline_sim(
    args: &BenchArgs,
    mode_name: &'static str,
    enable_fallback: bool,
) -> PipelineRunStats {
    let device = cuda_device();
    let cpu_dev = cpu_device();
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

    let cpu_model = switch_model_backend::<CudaBackend, CpuBackend>(model.clone(), &cpu_dev);
    let cpu_adj = match &cpu_model {
        HexGoModel::Gnn(_) => {
            let board = BoardDefinition::compact().graph().clone();
            adjacency_tensor::<CpuBackend>(&board, &cpu_dev)
        }
        _ => Tensor::zeros([1, 1], &cpu_dev),
    };

    let initial_games = args.infer_size.min(args.games);
    let count_0 = initial_games.div_ceil(2);
    let count_1 = initial_games - count_0;

    let mut slots = [GameSlot::new(count_0), GameSlot::new(count_1)];
    let mut started = slots[0].games.len() + slots[1].games.len();
    let mut completed = 0;
    let mut samples = Vec::new();

    let mut select_ns = 0u128;
    let mut forward_ns = 0u128;
    let mut update_ns = 0u128;
    let mut queue_wait_ns = 0u128;
    let mut commit_ns = 0u128;
    let mut forward_calls = 0u64;
    let mut gpu_forward_calls = 0u64;
    let mut cpu_forward_calls = 0u64;
    let mut total_evaluated_samples = 0u64;

    let overall_start = Instant::now();

    std::thread::scope(|s| {
        let (req_tx, req_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<Vec<f32>>)>(2);
        let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel(2);

        s.spawn(move || {
            while let Ok((id, inputs)) = req_rx.recv() {
                let num_samples = inputs.len();
                let is_gpu = !enable_fallback || num_samples >= GPU_MIN_BATCH;
                let t_start = Instant::now();
                let res = if is_gpu {
                    let refs: Vec<&Vec<f32>> = inputs.iter().collect();
                    forward_batch(&model, &device, &adj, &refs)
                } else {
                    forward_batch_cpu_parallel(&cpu_model, &cpu_dev, &cpu_adj, &inputs)
                };
                let duration_ns = t_start.elapsed().as_nanos();

                if resp_tx
                    .send((id, res, duration_ns, num_samples, is_gpu))
                    .is_err()
                {
                    break;
                }
            }
        });

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

        while in_flight > 0 {
            let t_wait = Instant::now();
            let (id, (policies, values), f_ns, num_samples, is_gpu) = resp_rx.recv().unwrap();
            queue_wait_ns += t_wait.elapsed().as_nanos();
            forward_ns += f_ns;
            forward_calls += 1;
            if is_gpu {
                gpu_forward_calls += 1;
            } else {
                cpu_forward_calls += 1;
            }
            total_evaluated_samples += num_samples as u64;
            in_flight -= 1;

            let slot = &mut slots[id];

            let t_up = Instant::now();
            slot.update(&policies, &values);
            update_ns += t_up.elapsed().as_nanos();

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

    PipelineRunStats {
        mode_name,
        overall_dur,
        completed,
        samples_len: samples.len(),
        forward_calls,
        gpu_forward_calls,
        cpu_forward_calls,
        total_evaluated_samples,
        select_ns,
        forward_ns,
        update_ns,
        queue_wait_ns,
        commit_ns,
    }
}

fn print_pipeline_breakdown(stats: &PipelineRunStats) {
    let total_ns = stats.overall_dur.as_nanos() as f64;
    let avg_batch_size = if stats.forward_calls > 0 {
        stats.total_evaluated_samples as f64 / stats.forward_calls as f64
    } else {
        0.0
    };
    let throughput = if stats.overall_dur.as_secs_f64() > 0.0 {
        stats.samples_len as f64 / stats.overall_dur.as_secs_f64()
    } else {
        0.0
    };

    println!("\n--- Pipeline Run Breakdown: {} ---", stats.mode_name);
    println!("Overall Time: {:.2?}", stats.overall_dur);
    println!(
        "Games: {} | Samples: {} | Throughput: {:.2} samples/sec",
        stats.completed, stats.samples_len, throughput
    );
    println!(
        "Total Forward Calls: {} (GPU >= 64: {}, CPU < 64: {}) | Avg Batch Size: {:.2}",
        stats.forward_calls, stats.gpu_forward_calls, stats.cpu_forward_calls, avg_batch_size
    );

    println!(
        "\n{:<35} {:<16} {:<18} {:<12}",
        "Stage", "Total Time (ms)", "Avg / Call (us)", "Percentage"
    );
    println!("--------------------------------------------------------------------------------");

    let print_row = |name: &str, ns: u128| {
        let ms = (ns as f64) / 1_000_000.0;
        let avg_us = if stats.forward_calls > 0 {
            (ns as f64) / (stats.forward_calls as f64 * 1000.0)
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

    print_row("1. CPU Select (Tree Search+Clone)", stats.select_ns);
    print_row("2. Forward (Inference + Transfer)", stats.forward_ns);
    print_row("3. CPU Update (Backpropagation)", stats.update_ns);
    print_row("4. Channel Wait (Queue Sync)", stats.queue_wait_ns);
    print_row("5. Commit Action (Move+Finish)", stats.commit_ns);
    println!("--------------------------------------------------------------------------------");
}

/// 3B. Benchmark CUDA batched pipeline with stage breakdown.
pub fn bench_cuda_pipeline_breakdown(args: &BenchArgs) {
    if args.games == 0 {
        println!("Skipping pipeline benchmark (games = 0).");
        return;
    }

    println!("\n================================================================================");
    println!("  [Benchmark 3/3] CUDA Batched Pipeline Stage Breakdown (Hybrid CPU Fallback)");
    println!("================================================================================");

    let stats = run_cuda_pipeline_sim(args, "Hybrid Pipeline", true);
    print_pipeline_breakdown(&stats);
}

/// 3C. A/B Benchmark directly comparing pipeline performance with vs without dynamic CPU fallback.
pub fn bench_pipeline_fallback_comparison(args: &BenchArgs) {
    println!("\n================================================================================");
    println!("  [A/B Benchmark] Impact Comparison: With vs Without Dynamic CPU Fallback");
    println!("================================================================================");
    println!(
        "Running comparison with {} games, {} iterations/move, infer batch size: {}...",
        args.games, args.iterations, args.infer_size
    );

    println!("\n[1/2] Running Baseline Pipeline WITHOUT Fallback (Pure GPU)...");
    let stats_without = run_cuda_pipeline_sim(args, "Without Fallback (Pure GPU)", false);
    print_pipeline_breakdown(&stats_without);

    println!("\n[2/2] Running Optimized Pipeline WITH Fallback (Hybrid Dynamic)...");
    let stats_with = run_cuda_pipeline_sim(args, "With Fallback (Hybrid Dynamic)", true);
    print_pipeline_breakdown(&stats_with);

    println!("\n================================================================================");
    println!("  [Executive Summary] Dynamic CPU Fallback Performance Impact");
    println!("================================================================================");
    println!(
        "{:<32} | {:<22} | {:<22} | {:<16}",
        "Metric", "Without Fallback", "With Fallback", "Improvement"
    );
    println!(
        "---------------------------------+------------------------+------------------------+------------------"
    );

    let t_without = stats_without.overall_dur.as_secs_f64();
    let t_with = stats_with.overall_dur.as_secs_f64();
    let speedup = if t_with > 0.0 {
        t_without / t_with
    } else {
        0.0
    };
    let time_saved_pct = if t_without > 0.0 {
        ((t_without - t_with) / t_without) * 100.0
    } else {
        0.0
    };

    println!(
        "{:<32} | {:<22.2?} | {:<22.2?} | {:<16}",
        "Total Wall-Clock Time",
        stats_without.overall_dur,
        stats_with.overall_dur,
        format!("{:.2}x ({:+.1}%)", speedup, -time_saved_pct)
    );

    let tput_without = if t_without > 0.0 {
        stats_without.samples_len as f64 / t_without
    } else {
        0.0
    };
    let tput_with = if t_with > 0.0 {
        stats_with.samples_len as f64 / t_with
    } else {
        0.0
    };
    let tput_boost = if tput_without > 0.0 {
        ((tput_with - tput_without) / tput_without) * 100.0
    } else {
        0.0
    };

    println!(
        "{:<32} | {:<22.2} | {:<22.2} | {:<16}",
        "Throughput (samples/sec)",
        tput_without,
        tput_with,
        format!("{:+.1}%", tput_boost)
    );

    println!(
        "{:<32} | {:<22} | {:<22} | {:<16}",
        "Total Forward Calls", stats_without.forward_calls, stats_with.forward_calls, "-"
    );

    let offloaded_pct = if stats_with.forward_calls > 0 {
        (stats_with.cpu_forward_calls as f64 / stats_with.forward_calls as f64) * 100.0
    } else {
        0.0
    };

    println!(
        "{:<32} | {:<22} | {:<22} | {:<16}",
        "  - GPU Calls (batch >= 64)",
        stats_without.gpu_forward_calls,
        stats_with.gpu_forward_calls,
        "-"
    );

    println!(
        "{:<32} | {:<22} | {:<22} | {:<16}",
        "  - CPU Fallback Calls (< 64)",
        stats_without.cpu_forward_calls,
        stats_with.cpu_forward_calls,
        format!("{:.1}% offloaded", offloaded_pct)
    );

    let f_ms_without = (stats_without.forward_ns as f64) / 1_000_000.0;
    let f_ms_with = (stats_with.forward_ns as f64) / 1_000_000.0;
    let f_saved_pct = if f_ms_without > 0.0 {
        ((f_ms_without - f_ms_with) / f_ms_without) * 100.0
    } else {
        0.0
    };

    println!(
        "{:<32} | {:<22.2} | {:<22.2} | {:<16}",
        "Forward Time (ms)",
        f_ms_without,
        f_ms_with,
        format!("{:+.1}%", -f_saved_pct)
    );

    println!("================================================================================\n");
}

/// Entry point to execute the appropriate benchmark suite based on CLI options.
pub fn run_benchmark(args: BenchArgs) {
    println!("Starting HexGo Benchmark Suite with config: {:?}", args);

    // 1. Benchmark pure board game rules (always runs)
    bench_game_logic();

    if args.compare_fallback {
        // Benchmark CUDA vs CPU parallel forward scaling across batch sizes
        bench_cuda_batch_scaling(args.model_type);
        // Run side-by-side A/B comparison of pipeline performance
        bench_pipeline_fallback_comparison(&args);
        return;
    }

    if args.compare {
        // Run both CPU and CUDA benchmarks side-by-side
        bench_cpu_independent_inference(args.model_type);
        bench_cuda_batch_scaling(args.model_type);
        bench_cpu_pipeline_breakdown(&args);
        bench_pipeline_fallback_comparison(&args);
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
            // Benchmark CUDA forward scaling across batch sizes (including CPU parallel comparison)
            bench_cuda_batch_scaling(args.model_type);
            // Benchmark CUDA double-buffered batched pipeline breakdown
            bench_cuda_pipeline_breakdown(&args);
            println!(
                "💡 Tip: Pass '--compare-fallback' to run side-by-side A/B comparison with vs without CPU dynamic fallback."
            );
        }
    }
}
