//! Batch processing example: create a batch, poll until complete, print results.
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example batch
//! ```

use std::time::Duration;

use claude_agent_rust_sdk::client::ClaudeClient;
use claude_agent_rust_sdk::types::batch::{BatchRequest, BatchResultBody, CreateBatchRequest};
use claude_agent_rust_sdk::types::{
    CreateMessageRequest, Message, MessageContent, Role,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY to run this example");

    let client = ClaudeClient::new(&api_key);

    // Build a batch of three translation requests.
    let languages = [("French", "Bonjour!"), ("Spanish", "Hola!"), ("German", "Guten Tag!")];

    let requests: Vec<BatchRequest> = languages
        .iter()
        .enumerate()
        .map(|(i, (lang, greeting))| BatchRequest {
            custom_id: format!("translate-{}", i),
            params: CreateMessageRequest {
                model: "claude-haiku-4-5".into(),
                max_tokens: 256,
                messages: vec![Message {
                    role: Role::User,
                    content: MessageContent::Text(format!(
                        "Translate this {} greeting to English and explain its cultural context: {}",
                        lang, greeting
                    )),
                }],
                system: None,
                temperature: None,
                top_p: None,
                stop_sequences: None,
                stream: None,
                tools: None,
                tool_choice: None,
                metadata: None,
                cache_control: None,
            },
        })
        .collect();

    println!("Creating batch with {} requests...", requests.len());
    let batch = client
        .batches()
        .create(&CreateBatchRequest { requests })
        .await?;

    println!("Batch ID: {}", batch.id);
    println!("Status:   {:?}", batch.processing_status);

    // Poll every 10 seconds until the batch finishes.
    println!("Polling for completion...");
    let completed = client
        .batches()
        .poll_until_complete(&batch.id, Duration::from_secs(10))
        .await?;

    println!(
        "Batch completed: {} succeeded, {} errored",
        completed.request_counts.succeeded, completed.request_counts.errored
    );

    // Fetch and print the results.
    let results = client.batches().results(&batch.id).await?;
    for result in &results {
        println!("\n--- {} ---", result.custom_id);
        match &result.result {
            BatchResultBody::Succeeded { message } => {
                println!("{}", message.text().unwrap_or("(no text)"));
            }
            BatchResultBody::Errored { error } => {
                println!("Error: {} - {}", error.error_type, error.message);
            }
            BatchResultBody::Canceled => println!("(canceled)"),
            BatchResultBody::Expired => println!("(expired)"),
        }
    }

    Ok(())
}
