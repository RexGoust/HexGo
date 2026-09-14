#[cfg(not(target_arch = "wasm32"))]
mod platform {
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread;

    type Job = Box<dyn FnOnce() + Send + 'static>;

    pub struct Future<T> {
        rx: Arc<Mutex<Receiver<T>>>,
    }

    impl<T> Future<T> {
        pub fn try_get(&self) -> Option<T> {
            self.rx.lock().unwrap().try_recv().ok()
        }
    }

    pub struct Worker {
        tx: Option<Sender<Job>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl Default for Worker {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Worker {
        pub fn new() -> Self {
            let (tx, rx) = mpsc::channel::<Job>();

            let handle = thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            });

            Worker {
                tx: Some(tx),
                handle: Some(handle),
            }
        }

        pub fn execute<F, T>(&self, f: F) -> Future<T>
        where
            F: FnOnce() -> T + Send + 'static,
            T: Send + 'static,
        {
            let (result_tx, result_rx) = mpsc::sync_channel(1);
            let job = Box::new(move || {
                let result = f();
                let _ = result_tx.send(result);
            });

            self.tx.as_ref().unwrap().send(job).unwrap();

            Future {
                rx: Arc::new(Mutex::new(result_rx)),
            }
        }
    }

    impl Drop for Worker {
        fn drop(&mut self) {
            self.tx.take();
            if let Some(h) = self.handle.take() {
                h.join().ok();
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod platform {
    use std::sync::{Arc, Mutex};

    pub struct Future<T> {
        result: Arc<Mutex<Option<T>>>,
    }

    impl<T> Future<T> {
        pub fn try_get(&self) -> Option<T> {
            self.result.lock().unwrap().take()
        }
    }

    #[derive(Default)]
    pub struct Worker;

    impl Worker {
        pub fn new() -> Self {
            Worker
        }

        pub fn execute<F, T>(&self, f: F) -> Future<T>
        where
            F: FnOnce() -> T + Send + 'static,
            T: Send + 'static,
        {
            let result = Arc::new(Mutex::new(None));
            let result_clone = Arc::clone(&result);

            wasm_bindgen_futures::spawn_local(async move {
                // Yield to allow the browser to paint the current frame
                // before running CPU-bound computation.
                yield_frame().await;

                let res = f();
                if let Ok(mut lock) = result_clone.lock() {
                    *lock = Some(res);
                }
            });

            Future { result }
        }
    }

    async fn yield_frame() {
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            if let Some(window) = web_sys::window() {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
            } else {
                let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
            }
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
}

pub use platform::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_execution() {
        let worker = Worker::new();
        let future = worker.execute(|| 42);

        for _ in 0..50 {
            if let Some(val) = future.try_get() {
                assert_eq!(val, 42);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        panic!("worker failed to produce result in time");
    }
}
