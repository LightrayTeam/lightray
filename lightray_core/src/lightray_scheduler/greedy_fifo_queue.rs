use crate::lightray_scheduler::queue::{LightrayWorkQueue, ResultWork};
use async_trait::async_trait;
use crossbeam_queue::SegQueue;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::oneshot::error::RecvError;
use tokio::sync::oneshot::{channel, Receiver, Sender};

struct ChannelBasedWork<T, Y> {
    payload: Arc<T>,
    sender: Sender<Y>,
}
pub struct LightrayFIFOWorkQueue<T, Y> {
    worker_queue: SegQueue<ChannelBasedWork<T, Y>>,
    phantom: PhantomData<Y>,
}

#[async_trait(?Send)]
impl<T, Y> LightrayWorkQueue<T, Y> for LightrayFIFOWorkQueue<T, Y> {
    async fn enqueue(&mut self, payload: Arc<T>) -> ResultWork<Y> {
        unimplemented!()
    }
    fn worker_loop(&mut self) {}
}
