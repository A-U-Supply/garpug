use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Vocab {
    pub char_to_idx: HashMap<char, u32>,
    pub idx_to_char: HashMap<u32, char>,
}

impl Vocab {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        chars.sort();
        let char_to_idx: HashMap<char, u32> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        let idx_to_char: HashMap<u32, char> = char_to_idx.iter()
            .map(|(&c, &i)| (i, c))
            .collect();
        Self { char_to_idx, idx_to_char }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.idx_to_char.get(&id))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.char_to_idx.len()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let vocab: Self = serde_json::from_str(&json)?;
        Ok(vocab)
    }
}

/// Fetch text from a file path or URL
pub fn fetch_text(input: &str) -> Result<String> {
    if input.starts_with("http://") || input.starts_with("https://") {
        let resp = reqwest::blocking::get(input)
            .context("Failed to fetch URL")?;
        let text = resp.text().context("Failed to read response body")?;
        Ok(text)
    } else {
        std::fs::read_to_string(input)
            .context(format!("Failed to read file: {input}"))
    }
}

/// Strip Project Gutenberg header and footer boilerplate
pub fn strip_gutenberg(text: &str) -> &str {
    let start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ];
    let end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ];

    let start = start_markers.iter()
        .filter_map(|marker| text.find(marker))
        .min()
        .map(|pos| {
            // Skip past the marker line
            text[pos..].find('\n').map(|nl| pos + nl + 1).unwrap_or(pos)
        })
        .unwrap_or(0);

    let end = end_markers.iter()
        .filter_map(|marker| text[start..].find(marker).map(|p| p + start))
        .min()
        .unwrap_or(text.len());

    text[start..end].trim()
}

pub struct Dataset {
    pub train: Vec<u32>,
    pub val: Vec<u32>,
}

impl Dataset {
    pub fn from_tokens(tokens: Vec<u32>, val_fraction: f32) -> Self {
        let split = ((tokens.len() as f32) * (1.0 - val_fraction)) as usize;
        let train = tokens[..split].to_vec();
        let val = tokens[split..].to_vec();
        Self { train, val }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_roundtrip() {
        let vocab = Vocab::from_text("hello world");
        let encoded = vocab.encode("hello");
        let decoded = vocab.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_vocab_size() {
        let vocab = Vocab::from_text("aabbc");
        assert_eq!(vocab.size(), 3); // a, b, c
    }

    #[test]
    fn test_strip_gutenberg() {
        let text = "Some header stuff\n*** START OF THIS PROJECT GUTENBERG EBOOK ***\nActual book content here.\n*** END OF THIS PROJECT GUTENBERG EBOOK ***\nSome footer.";
        let stripped = strip_gutenberg(text);
        assert_eq!(stripped, "Actual book content here.");
    }

    #[test]
    fn test_strip_gutenberg_no_markers() {
        let text = "Just plain text with no markers.";
        let stripped = strip_gutenberg(text);
        assert_eq!(stripped, "Just plain text with no markers.");
    }

    #[test]
    fn test_dataset_split() {
        let tokens: Vec<u32> = (0..100).collect();
        let ds = Dataset::from_tokens(tokens, 0.1);
        assert_eq!(ds.train.len(), 90);
        assert_eq!(ds.val.len(), 10);
    }

    #[test]
    fn test_fetch_text_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("garpug_test.txt");
        std::fs::write(&path, "test content").unwrap();
        let result = fetch_text(path.to_str().unwrap()).unwrap();
        assert_eq!(result, "test content");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_vocab_save_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("garpug_vocab_test.json");
        let vocab = Vocab::from_text("abc");
        vocab.save(&path).unwrap();
        let loaded = Vocab::load(&path).unwrap();
        assert_eq!(loaded.size(), 3);
        assert_eq!(loaded.encode("abc"), vocab.encode("abc"));
        std::fs::remove_file(&path).ok();
    }
}
