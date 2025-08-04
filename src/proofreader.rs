use gemini_rs::prelude::{
    Content, GeminiClient, GenerateContentRequest, GenerationConfig, Part, Role, TokenProvider,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const SYSTEM_PROMPT: &str = r" \
You are an expert proofreader. Analyze the provided text and identify any corrections needed to fix
grammar or spelling. For each mistake, identify the startIndex, endIndex, the mistake type, report
the correction and an explanation. The startIndex must be the first character of the error in the
original text and endIndex the last. IT'S VERY IMPORTANT TO GET startIndex AND endIndex CORRECTLY,
OTHERWISE KITTENS WILL DIE. Finally, provide the entire corrected string.";
const MODEL: &str = "gemini-2.0-pro-exp-02-05";

#[derive(Serialize, Deserialize)]
pub struct Proofreading {
    pub corrected: String,
    pub corrections: Vec<Correction>,
}

#[derive(Serialize, Deserialize)]
pub struct Correction {
    #[serde(rename = "startIndex")]
    pub start_index: usize,
    #[serde(rename = "endIndex")]
    pub end_index: usize,
    pub correction: String,
    #[serde(rename = "type")]
    pub correction_type: CorrectionType,
    pub explanation: String,
}

#[derive(Serialize, Deserialize)]
pub enum CorrectionType {
    #[serde(rename = "spelling")]
    Spelling,
    #[serde(rename = "punctuation")]
    Punctuation,
    #[serde(rename = "capitalization")]
    Capitalization,
    #[serde(rename = "preposition")]
    Preposition,
    #[serde(rename = "missing-words")]
    MissingWords,
    #[serde(rename = "grammar")]
    Grammar,
}

pub async fn proofread<T: TokenProvider + Clone>(
    vertex_client: &GeminiClient<T>,
    input: &str,
) -> gemini_rs::error::Result<Proofreading> {
    let request = GenerateContentRequest {
        system_instruction: Some(Content {
            parts: Some(vec![Part::Text(SYSTEM_PROMPT.to_string())]),
            role: None,
        }),
        generation_config: Some(GenerationConfig {
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(json!({
              "type": "object",
              "properties" :{
                "corrected": {
                  "type": "string"
                },
                "corrections": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "startIndex": {
                        "type": "integer"
                      },
                      "endIndex": {
                        "type": "integer"
                      },
                      "correction": {
                        "type": "string"
                      },
                      "type": {
                        "type": "string",
                        "enum": ["spelling", "punctuation", "capitalization", "preposition", "missing-words", "grammar"]
                      },
                      "explanation" : {
                        "type": "string"
                      }
                    },
                    "required": ["startIndex", "endIndex", "correction", "type", "explanation"]
                  }
                }
              }
            })),
            ..Default::default()
        }),
        contents: vec![Content {
            parts: Some(vec![Part::Text(input.to_string())]),
            role: Some(Role::User),
        }],
        ..Default::default()
    };
    let response = vertex_client.generate_content(&request, MODEL).await?;
    let result = response.candidates[0].get_text().unwrap_or_default();
    let result: Proofreading = serde_json::from_str(&result)?;
    Ok(result)
}
