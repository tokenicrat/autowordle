# What a nice script. Thanks Claude

import json
import argparse
import os

def filter_words(input_file, output_file, word_length=5):
    """
    Reads words from a JSON file, filters by length, and writes to a new JSON file.
    
    Args:
        input_file (str): Path to the input JSON file containing an array of words
        output_file (str): Path to save the filtered words as a JSON array
        word_length (int): The character length to filter for (default: 5)
    """
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            words = json.load(f)
        
        # Validate input format
        if not isinstance(words, list):
            raise TypeError("Input JSON must be an array of words")
        
        # Filter words by length
        filtered_words = [word for word in words if isinstance(word, str) and len(word) == word_length]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write filtered words to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_words, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully filtered {len(filtered_words)} words of length {word_length} to {output_file}")
        print(f"Original file had {len(words)} words total")
        
    except json.JSONDecodeError:
        print(f"Error: {input_file} contains invalid JSON")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except Exception as e:
        print(f"Error: {e}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Filter words in a JSON array by character length')
    parser.add_argument('input_file', help='Path to input JSON file containing an array of words')
    parser.add_argument('output_file', help='Path to output filtered JSON file')
    parser.add_argument('-l', '--length', type=int, default=5, 
                        help='Filter words of this character length (default: 5)')
    
    args = parser.parse_args()
    
    # Call the filter function with provided arguments
    filter_words(args.input_file, args.output_file, args.length)

if __name__ == "__main__":
    main()
