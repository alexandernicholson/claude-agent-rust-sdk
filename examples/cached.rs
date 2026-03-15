//! Prompt caching example: attach cache-control directives to system prompts
//! and messages.
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example cached
//! ```

use claude_agent_rust_sdk::client::ClaudeClient;
use claude_agent_rust_sdk::types::CacheControl;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY to run this example");

    let client = ClaudeClient::new(&api_key);

    // A large system prompt that benefits from caching.  In production this
    // might be a multi-page style guide, knowledge base, or specification.
    let large_context = "You are an expert legal assistant. \
        You have memorized the entire United States Code, all federal \
        regulations, and case law from the Supreme Court through 2024. \
        When answering questions, cite the relevant statute or case. \
        Always provide a balanced analysis of both sides of an argument. \
        Format citations in Bluebook style."
        .repeat(10); // Repeat to simulate a large context

    let response = client
        .messages()
        .model("claude-haiku-4-5")
        .max_tokens(1024)
        // This large context block will be cached for 5 minutes.
        .system_with_cache(&large_context, CacheControl::ephemeral_5m())
        .user("What is the legal standard for negligence in tort law?")
        .temperature(0.3)
        .send()
        .await?;

    println!("Model: {}", response.model);
    println!(
        "Usage: {} input / {} output tokens",
        response.usage.input_tokens, response.usage.output_tokens
    );
    if let Some(cached) = response.usage.cache_read_input_tokens {
        println!("Cache read tokens: {}", cached);
    }
    if let Some(created) = response.usage.cache_creation_input_tokens {
        println!("Cache creation tokens: {}", created);
    }
    println!();
    println!("{}", response.text().unwrap_or("(no text in response)"));

    // Send a second request -- the system prompt should be served from cache.
    println!("\n--- Second request (should hit cache) ---\n");

    let response2 = client
        .messages()
        .model("claude-haiku-4-5")
        .max_tokens(1024)
        .system_with_cache(&large_context, CacheControl::ephemeral_5m())
        .user("What is the difference between a felony and a misdemeanor?")
        .temperature(0.3)
        .send()
        .await?;

    println!(
        "Usage: {} input / {} output tokens",
        response2.usage.input_tokens, response2.usage.output_tokens
    );
    if let Some(cached) = response2.usage.cache_read_input_tokens {
        println!("Cache read tokens: {}", cached);
    }
    println!();
    println!("{}", response2.text().unwrap_or("(no text in response)"));

    Ok(())
}
