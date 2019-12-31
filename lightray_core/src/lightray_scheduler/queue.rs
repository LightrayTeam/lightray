use crate::lightray_scheduler::metrics::SchedulerMetric;
use async_trait::async_trait;
use std::future::Future;
use std::sync::Arc;
pub struct ResultWork<T> {
    payload: Arc<T>,
    error: Option<Arc<dyn std::error::Error>>,
    metrics: SchedulerMetric,
}
#[async_trait(?Send)]
pub trait LightrayWorkQueue<T, Y> {
    async fn enqueue(&mut self, payload: Arc<T>) -> ResultWork<Y>;
    fn worker_loop(&mut self);
}
