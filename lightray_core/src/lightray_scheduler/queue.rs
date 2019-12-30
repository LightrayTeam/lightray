use std::future::Future;
use std::sync::Arc;
pub struct ResultWork<T> {
    payload: Arc<T>,
}
pub trait LightrayWorkQueue<T> {
    fn enqueue(&mut self, payload: T) -> Future<Output = ResultWork<T>>;
}
