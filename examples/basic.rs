//! Basic example: send a single message and print the response.
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example basic
//! ```

use claude_agent_rust_sdk::client::ClaudeClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise tracing so SDK debug logs are visible when RUST_LOG is set.
    tracing_subscriber::fmt::init();

    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY to run this example");

    let client = ClaudeClient::new(&api_key);

    let response = client
        .messages()
        .model("claude-haiku-4-5")
        .max_tokens(1024)
        .system("You are a friendly assistant. Keep answers brief.")
        .user("What are three interesting facts about Rust the programming language?")
        .temperature(0.7)
        .send()
        .await?;

    println!("Model: {}", response.model);
    println!("Stop reason: {:?}", response.stop_reason);
    println!(
        "Usage: {} input / {} output tokens",
        response.usage.input_tokens, response.usage.output_tokens
    );
    println!();
    println!("{}", response.text().unwrap_or("(no text in response)"));

    Ok(())
}
