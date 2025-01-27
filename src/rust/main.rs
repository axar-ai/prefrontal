use prefrontal::{Classifier, BuiltinModel, ClassDefinition, ModelManager};
use log::info;
use env_logger;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Force a fresh download of the model files
    #[arg(short, long)]
    fresh: bool,
}

async fn ensure_model_downloaded(fresh: bool) -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;
    let model_info = BuiltinModel::MiniLM.get_model_info();

    if fresh {
        info!("Fresh download requested - removing any existing model files...");
        manager.remove_download(&model_info.name)?;
    }
    
    manager.ensure_model_downloaded(&model_info).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    
    info!("=== Starting Text Classifier Demo ===");

    // Ensure model is downloaded before proceeding
    ensure_model_downloaded(args.fresh).await?;

    let start_time = std::time::Instant::now();
    info!("Building classifier with {} classes...", 5);
    
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities including team and individual sports"
            ).with_examples(vec![
                // Team Sports
                "The team scored in the final minutes of the championship game",
                "Local basketball team advances to state finals",
                "Soccer match ends in dramatic penalty shootout",
                "Baseball team clinches division title with walk-off home run",
                "Hockey team wins overtime thriller on power play goal",
                "Rugby team dominates in international tournament",
                "Volleyball team secures spot in national championships",
                
                // Individual Sports
                "Tennis player wins her third grand slam title",
                "Gymnast performs perfect routine at Olympics",
                "Swimmer breaks world record in freestyle event",
                "Boxer wins heavyweight championship bout",
                "Golfer sinks dramatic putt to win major tournament"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "tech",
                "Technology-related content including software, hardware, and scientific advances"
            ).with_examples(vec![
                // Software & Programming
                "New programming language features improve developer productivity",
                "Latest software update includes security fixes",
                "Open source project reaches major milestone",
                "Cloud computing platform announces new services",
                
                // Hardware & Devices
                "Smartphone manufacturer reveals next-gen device",
                "New processor architecture promises efficiency gains",
                "Gaming console sets sales records",
                
                // AI & Machine Learning
                "AI model achieves human-level performance",
                "Machine learning algorithm improves accuracy",
                "Neural network breakthrough in image recognition"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "entertainment",
                "Entertainment and media content including movies, music, and television"
            ).with_examples(vec![
                // Movies
                "Blockbuster movie breaks opening weekend records",
                "Director announces sequel to popular film",
                "Actor wins award for dramatic performance",
                
                // Music
                "Band releases highly anticipated album",
                "Singer announces world tour dates",
                "Music festival lineup revealed",
                
                // Television
                "TV series finale draws record viewers",
                "Streaming service launches original content",
                "Reality show contestant wins competition"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "business",
                "Business and financial news including corporate events and market updates"
            ).with_examples(vec![
                // Corporate News
                "Company reports record quarterly earnings",
                "Startup raises significant funding round",
                "CEO announces strategic restructuring",
                
                // Market News
                "Stock market reaches new high",
                "Currency trading shows increased volatility",
                "Commodity prices affect global markets",
                
                // Industry Trends
                "Retail sector shows strong growth",
                "Manufacturing output exceeds expectations",
                "Tech industry faces new regulations"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "science",
                "Scientific discoveries and research across various fields"
            ).with_examples(vec![
                // Physics & Space
                "Astronomers discover new exoplanet",
                "Particle accelerator yields unexpected results",
                "Space telescope captures distant galaxy",
                
                // Biology & Medicine
                "Researchers identify new species",
                "Clinical trial shows promising results",
                "Genetic study reveals evolutionary insights",
                
                // Environmental Science
                "Climate study predicts future trends",
                "Ocean research finds new phenomena",
                "Geological survey maps fault lines"
            ])
        )?
        .build()?;

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
