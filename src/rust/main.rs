use prefrontal::{Classifier, BuiltinModel, ClassDefinition, ModelManager};
use log::info;
use env_logger;
use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Force a fresh download of the model files
    #[arg(short, long)]
    fresh: bool,
}

async fn ensure_model_downloaded(fresh: bool) -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;
    let model = BuiltinModel::MiniLM;

    if fresh {
        info!("Fresh download requested - removing any existing model files...");
        manager.remove_download(model)?;
    }
    
    if !manager.is_model_downloaded(model) {
        info!("Downloading model...");
        manager.download_model(model).await?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    
    info!("=== Starting Text Classifier Demo ===");

    // Ensure model is downloaded before proceeding
    ensure_model_downloaded(args.fresh).await?;

    let start_time = Instant::now();
    info!("Building classifier...");

    // Create classifier
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new("positive", "Content with positive sentiment")
                .with_examples(vec!["great", "awesome", "excellent"])
        )?
        .add_class(
            ClassDefinition::new("negative", "Content with negative sentiment")
                .with_examples(vec!["bad", "terrible", "awful"])
        )?
        .build()?;

    // Test classification
    let test_text = "This is a great movie!";
    info!("Testing classification with text: {}", test_text);
    let (label, scores) = classifier.predict(test_text)?;
    println!("Predicted class: {}", label);
    println!("Confidence scores: {:?}", scores);

    let build_time = start_time.elapsed();
    info!("=== Classifier Built Successfully (took {:.2?}) ===\n", build_time);
    
    let test_inputs = vec![
        // Clear single-category cases
        "The new AI model shows remarkable accuracy in natural language processing tasks",
        "Team wins championship in dramatic overtime thriller with last-second goal",
        "Movie premiere attracts A-list celebrity audience at red carpet event",
        "Company stock surges 20% after exceeding quarterly earnings expectations",
        "Scientists discover evidence of ancient microbial life on Mars",

        // Mixed-category cases
        "Tech company develops AI for analyzing sports performance",
        "Business invests heavily in entertainment streaming platform",
        "Scientific study on economic impact of sports industry",

        // Edge cases
        "This is a very short text",
        "This text doesn't clearly belong to any specific category but should still be classified",
        "Breaking: Major announcement today about multiple sectors including technology, business, and science",
    ];

    info!("=== Running Classifications ({} inputs) ===\n", test_inputs.len());
    let classify_start = std::time::Instant::now();
    
    for (i, text) in test_inputs.iter().enumerate() {
        info!("\nTest {}/{} (elapsed: {:.2?}):", i + 1, test_inputs.len(), classify_start.elapsed());
        info!("Input: {}", text);
        process_input(&classifier, text)?;
    }

    let total_time = start_time.elapsed();
    let classify_time = classify_start.elapsed();
    
    info!("\n=== Demo Complete ===");
    info!("Total time: {:.2?}", total_time);
    info!("Build time: {:.2?}", build_time);
    info!("Classification time: {:.2?}", classify_time);
    info!("Average time per classification: {:.2?}", classify_time / test_inputs.len() as u32);
    
    Ok(())
}

fn process_input(classifier: &Classifier, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("\nProcessing: {}", text);
    
    match classifier.predict(text) {
        Ok((class, scores)) => {
            let mut scores: Vec<_> = scores.into_iter().collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            println!("\nResults:");
            println!("  Predicted class: {}", class);
            println!("  Confidence scores (sorted):");
            for (label, score) in scores {
                println!("    {}: {:.1}%", label, score * 100.0);
            }
        }
        Err(e) => {
            eprintln!("\nError processing text: {}", e);
            eprintln!("Consider:");
            eprintln!("  - Checking if the text is empty");
            eprintln!("  - Splitting long text into smaller chunks (max 256 tokens)");
            eprintln!("  - Ensuring the text is valid UTF-8");
            return Err(e.into());
        }
    }

    Ok(())
}
