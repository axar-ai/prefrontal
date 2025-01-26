use text_classifier::{Classifier, BuiltinModel, ClassDefinition};
use log::info;
use env_logger;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("=== Starting Text Classifier Demo ===");

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

    info!("=== Classifier Built Successfully ===\n");
    
    let test_inputs = vec![
        "The new AI model shows remarkable accuracy in predictions",
        "Team wins championship in overtime thriller",
        "Movie premiere attracts celebrity audience",
        "Company stock surges after earnings report",
        "Scientists discover evidence of ancient life",
    ];

    info!("=== Running Classifications ===\n");
    for (i, text) in test_inputs.iter().enumerate() {
        info!("\nTest {}/{}:", i + 1, test_inputs.len());
        info!("Input: {}", text);
        process_input(&classifier, text)?;
    }

    info!("\n=== Demo Complete ===\n");
    Ok(())
}

fn process_input(classifier: &Classifier, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nProcessing: {}", text);
    
    let (class, scores) = classifier.predict(text)?;
    
    println!("Predicted class: {}", class);
    println!("Confidence scores:");
    for (label, score) in scores {
        println!("  {}: {:.4}", label, score);
    }

    Ok(())
}
