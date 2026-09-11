//! The async side: a run turned into a [`Stream`] of chunks.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;
use rkllm_sys::RkllmApi;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::error::{Error, Result};
use crate::infer::InferParams;
use crate::input::Input;
use crate::output::{Chunk, Control};
use crate::session::RkllmSession;

/// How many chunks the channel holds before the runtime is made to wait.
const DEFAULT_BUFFER: usize = 64;

impl<A> RkllmSession<A>
where
    A: RkllmApi + Send + Sync + 'static,
{
    /// Runs inference on a blocking thread and streams the chunks back.
    ///
    /// This is [`RkllmSession::run_llm`] with a channel for a callback. The
    /// closure hands each chunk to a `tokio::sync::mpsc` sender, and the
    /// receiver becomes the returned stream.
    ///
    /// Dropping the stream cancels the run. The next chunk the runtime produces
    /// finds the receiver gone, and generation stops there rather than at the
    /// end of the token budget.
    ///
    /// Only text, token ids and perf stats survive the trip. Hidden states and
    /// logits are left behind, since they are large and borrowed. Use
    /// [`RkllmSession::run_llm`] directly for those.
    ///
    /// ```no_run
    /// # use futures_util::StreamExt as _;
    /// # use rkllm::{InferParams, Input, RkllmSession, Result};
    /// # use std::sync::Arc;
    /// # use rkllm_sys::RkllmApi;
    /// # async fn f<A>(session: &Arc<RkllmSession<A>>) -> Result<()>
    /// # where A: RkllmApi + Send + Sync + 'static {
    /// let mut stream = session.run_llm_async(Input::prompt("Hello")?, InferParams::new());
    /// while let Some(chunk) = stream.next().await {
    ///     print!("{}", chunk?.text());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_llm_async(self: &Arc<Self>, input: Input, params: InferParams) -> ChunkStream {
        self.run_llm_async_buffered(input, params, DEFAULT_BUFFER)
    }

    /// [`RkllmSession::run_llm_async`], with the channel capacity spelled out.
    ///
    /// A capacity of zero is treated as one, since the channel needs room for
    /// at least one chunk.
    pub fn run_llm_async_buffered(
        self: &Arc<Self>,
        mut input: Input,
        params: InferParams,
        buffer: usize,
    ) -> ChunkStream {
        let (sender, receiver) = mpsc::channel(buffer.max(1));
        let session = Arc::clone(self);

        let task = tokio::task::spawn_blocking(move || {
            session.run_llm(&mut input, &params, |output| {
                match sender.blocking_send(output.to_chunk()) {
                    Ok(()) => Control::Continue,
                    // The stream was dropped, so nobody is left to read this.
                    Err(_) => Control::Pause,
                }
            })
        });

        ChunkStream {
            receiver,
            task: Some(task),
        }
    }
}

/// The chunks of one run, in order, ending with whatever the run returned.
///
/// Yields `Ok` for each chunk. If the run itself failed, or the blocking task
/// panicked, that arrives as a single `Err` after the last chunk.
///
/// Dropping this cancels the run.
#[derive(Debug)]
pub struct ChunkStream {
    receiver: mpsc::Receiver<Chunk>,
    task: Option<JoinHandle<Result<()>>>,
}

impl ChunkStream {
    /// Stops accepting chunks, which cancels the run at its next output.
    pub fn close(&mut self) {
        self.receiver.close();
    }
}

impl Stream for ChunkStream {
    type Item = Result<Chunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        match this.receiver.poll_recv(cx) {
            Poll::Ready(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
            Poll::Pending => return Poll::Pending,
            // The sender is gone, so the blocking task has finished. Its return
            // value is the run's own result, which is the last thing to report.
            Poll::Ready(None) => {}
        }

        let Some(task) = this.task.as_mut() else {
            return Poll::Ready(None);
        };

        match Pin::new(task).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(joined) => {
                this.task = None;
                match joined {
                    Ok(Ok(())) => Poll::Ready(None),
                    Ok(Err(run)) => Poll::Ready(Some(Err(run))),
                    Err(join) => Poll::Ready(Some(Err(Error::Task(join)))),
                }
            }
        }
    }
}
