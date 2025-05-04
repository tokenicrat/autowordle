import asyncio
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum, auto
import logging

from playwright.async_api import async_playwright

# Colorized terminal output
class Colors:
    INFO = '\033[94m'     # Blue
    WARNING = '\033[93m'  # Yellow
    ERROR = '\033[91m'    # Red
    SUCCESS = '\033[92m'  # Green
    RESET = '\033[0m'     # Reset

# Custom logging formatter to colorize log levels
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname == 'INFO':
            record.levelname = f"{Colors.INFO}{levelname}{Colors.RESET}"
        elif levelname == 'WARNING':
            record.levelname = f"{Colors.WARNING}{levelname}{Colors.RESET}"
        elif levelname == 'ERROR':
            record.levelname = f"{Colors.ERROR}{levelname}{Colors.RESET}"
        elif levelname == 'SUCCESS':  # Custom level
            record.levelname = f"{Colors.SUCCESS}SUCCESS{Colors.RESET}"
        return super().format(record)

# Add custom SUCCESS level
SUCCESS_LEVEL = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(levelname)s - %(message)s'))
logger = logging.getLogger('wordle_solver')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Clear default handlers if any
logger.propagate = False

# Add success method to logger
def success(self, message, *args, **kwargs):
    self.log(SUCCESS_LEVEL, message, *args, **kwargs)
    
logging.Logger.success = success

# Game constants
WORDLE_URL = 'https://wordly.org/'
MAX_ATTEMPTS = 6
WORD_LENGTH = 5

# Browser configuration
LOCAL_CHROME_PATH = '/opt/google/chrome/google-chrome'  # Set to None to use default Chrome

class LetterState(Enum):
    """Enum for letter evaluation states in Wordle"""
    CORRECT = auto()    # Letter is in the correct position
    ELSEWHERE = auto()  # Letter is in the word but in the wrong position
    ABSENT = auto()     # Letter is not in the word at all

@dataclass
class WordleState:
    """Class to track the state of a Wordle game"""
    # Map of letters to their status information
    # The tuple contains (correct_positions, wrong_positions, state)
    letter_status: Dict[str, Tuple[List[int], List[int], LetterState]]
    
    # Map of positions to confirmed letters
    known_positions: Dict[int, str]
    
    # Set of previously guessed words
    guessed_words: Set[str]
    
    def __init__(self):
        self.letter_status = {}
        self.known_positions = {}
        self.guessed_words = set()

class WordleSolver:
    """Class for solving Wordle puzzles"""
    
    def __init__(self, word_database: List[Dict[str, Any]]):
        """Initialize with a database of words and their weights"""
        self.words = word_database
        self.state = WordleState()
        
    def evaluate_word(self, word: str) -> float:
        """Calculate a score for a word based on the current game state"""
        if not word or len(word) != WORD_LENGTH:
            return 0.0
        
        # Skip words we've already guessed
        if word in self.state.guessed_words:
            return 0.0
        
        # Check if word meets our known constraints
        if not self._word_matches_constraints(word):
            return 0.0
        
        # Get the word's weight from our database
        for word_obj in self.words:
            if word_obj['word'] == word:
                # Force diversity in the first few guesses by overriding normal weights
                # This prevents the algorithm from always choosing the same high-weight word
                if len(self.state.guessed_words) == 0:
                    # For first guess, letter uniqueness is far more important than dictionary weight
                    unique_letters = len(set(word))
                    return unique_letters * 2.0 + float(word_obj['weight']) * 0.1
                
                # Apply additional heuristics based on game progress
                base_weight = float(word_obj['weight'])
                return self._apply_heuristics(word, base_weight)
                
        return 0.0
    
    def _word_matches_constraints(self, word: str) -> bool:
        """Check if a word matches all our known constraints"""
        # Skip empty check if we have no constraints yet
        if not self.state.known_positions and not self.state.letter_status:
            return True
            
        # Check known positions
        for pos, char in self.state.known_positions.items():
            if word[pos] != char:
                return False
        
        # Check letter status constraints
        for char, (correct_pos, wrong_pos, state) in self.state.letter_status.items():
            char_positions = [i for i, c in enumerate(word) if c == char]
            
            # If letter is marked as absent and appears in word
            if state == LetterState.ABSENT and char_positions:
                # Exception: if the letter appears in a correct position elsewhere,
                # it might be a duplicate letter scenario
                if not char in self.state.known_positions.values():
                    return False
            
            # If letter should be in specific positions
            if correct_pos and not all(pos in char_positions for pos in correct_pos):
                return False
            
            # If letter should not be in certain positions
            if wrong_pos and any(pos in char_positions for pos in wrong_pos):
                return False
            
            # If we know letter exists but word doesn't contain it
            if (state in [LetterState.CORRECT, LetterState.ELSEWHERE]) and not char_positions:
                return False
                
        return True
    
    def _apply_heuristics(self, word: str, base_weight: float) -> float:
        """Apply additional scoring heuristics based on game progress"""
        known_chars = {
            char for char, (_, _, state) in self.state.letter_status.items() 
            if state in [LetterState.CORRECT, LetterState.ELSEWHERE]
        }
        
        # First guess strategy: significantly boost words with unique letters
        # to diversify information gathering, regardless of weight
        if len(self.state.guessed_words) == 0:
            # First guess should prioritize high-information words
            unique_letters = len(set(word))
            
            # For first guess, unique letters are more important than word weight
            # This prevents always picking the highest weighted word
            if unique_letters == WORD_LENGTH:
                # Strongly favor words with 5 unique letters for first guess
                return base_weight * 10.0 + 5.0
            else:
                # Slightly lower score for words with duplicate letters
                return base_weight * 5.0 + unique_letters
        
        # Early game strategy (guesses 2-3): prioritize information gathering
        elif len(self.state.guessed_words) <= 2:
            # Get all letters we've already tried
            tried_letters = set(''.join(self.state.guessed_words))
            
            # Strongly favor words with new letters we haven't tried yet
            new_letters = sum(1 for c in word if c not in tried_letters)
            information_factor = 1.0 + (new_letters * 0.5)
            
            # Penalize words containing letters we know are absent
            absent_letters = {
                char for char, (_, _, state) in self.state.letter_status.items()
                if state == LetterState.ABSENT
            }
            absent_count = sum(1 for c in word if c in absent_letters)
            if absent_count > 0:
                information_factor *= 0.3
            
            # Slightly increase weight for words with some known letters
            if known_chars:
                known_count = sum(1 for c in word if c in known_chars)
                if 0 < known_count <= 2:
                    information_factor *= 1.2
            
            return base_weight * information_factor
        
        # Mid-game strategy (guesses 3-4): balance between using known letters and finding new ones
        elif len(self.state.guessed_words) < 4:
            # Balance between known information and exploring new letters
            if known_chars:
                # Prioritize using known letters, but not exclusively
                known_count = sum(1 for c in word if c in known_chars)
                balance_factor = 1.0 + (known_count * 0.3)
                
                # Slightly favor words that also introduce new letters
                tried_letters = set(''.join(self.state.guessed_words))
                new_letters = sum(1 for c in word if c not in tried_letters)
                if new_letters > 0:
                    balance_factor *= (1.0 + (new_letters * 0.1))
                    
                return base_weight * balance_factor
            return base_weight * 1.2
        
        # Late game strategy (guesses 5-6): focus entirely on likely solutions
        else:
            # At this point, prioritize words that use our known information
            solution_factor = 1.0
            
            # Heavily prioritize words with known correct/elsewhere letters
            if known_chars:
                known_count = sum(1 for c in word if c in known_chars)
                solution_factor *= (1.0 + (known_count * 0.5))
                
            # Boost words with common letter patterns
            common_patterns = ['th', 'er', 'on', 'an', 'in', 'es', 'ar', 'te', 'al', 'ed']
            for pattern in common_patterns:
                if pattern in word:
                    solution_factor *= 1.1
                    
            # Consider letter position frequencies in English
            common_first_letters = 'stcpabmdf'
            common_last_letters = 'estnrlyod'
            if word[0] in common_first_letters:
                solution_factor *= 1.2
            if word[-1] in common_last_letters:
                solution_factor *= 1.2
                    
            return base_weight * solution_factor
        
    def suggest_next_word(self) -> Optional[str]:
        """Select the best word to guess next based on current game state"""
        best_word = None
        best_score = -1
        
        # Keep track of the top 5 candidates for debugging/transparency
        top_candidates = []
        
        for word_obj in self.words:
            word = word_obj['word']
            score = self.evaluate_word(word)
            
            # Track top candidates
            if score > 0:
                top_candidates.append((word, score))
                top_candidates.sort(key=lambda x: x[1], reverse=True)
                if len(top_candidates) > 5:
                    top_candidates.pop()
            
            if score > best_score:
                best_score = score
                best_word = word
        
        if best_word:
            self.state.guessed_words.add(best_word)
            
            # Log top candidates for transparency
            if top_candidates:
                logger.info(f"Top candidates: {', '.join([f'{w}({s:.2f})' for w, s in top_candidates])}")
            
        return best_word
        
    def update_state(self, guess: str, feedback: List[Tuple[str, LetterState]]) -> bool:
        """Update game state based on feedback from the last guess
        
        Returns True if the word is solved (all letters correct)
        """
        word_complete = True
        
        for idx, (letter, state) in enumerate(feedback):
            # Update known positions for correct letters
            if state == LetterState.CORRECT:
                self.state.known_positions[idx] = letter
            else:
                word_complete = False
                
            # Update letter status
            if letter not in self.state.letter_status:
                self.state.letter_status[letter] = ([], [], state)
            
            correct_pos, wrong_pos, current_state = self.state.letter_status[letter]
            
            if state == LetterState.CORRECT:
                if idx not in correct_pos:
                    correct_pos.append(idx)
                # Upgrade state to CORRECT if it was ELSEWHERE before
                self.state.letter_status[letter] = (correct_pos, wrong_pos, LetterState.CORRECT)
                
            elif state == LetterState.ELSEWHERE:
                if idx not in wrong_pos:
                    wrong_pos.append(idx)
                # Only update state if it wasn't already CORRECT
                if current_state != LetterState.CORRECT:
                    self.state.letter_status[letter] = (correct_pos, wrong_pos, LetterState.ELSEWHERE)
                    
            elif state == LetterState.ABSENT:
                # Only set to ABSENT if we don't already know it's correct or elsewhere
                if current_state not in [LetterState.CORRECT, LetterState.ELSEWHERE]:
                    self.state.letter_status[letter] = (correct_pos, wrong_pos, LetterState.ABSENT)
                    
        return word_complete

async def run_wordle_solver():
    # Load word database once at start
    try:
        with open('worddb/weighted_words.json', 'r') as f:
            words = json.load(f)
        logger.info(f"Loaded {len(words)} words from database")
    except Exception as e:
        logger.error(f"Error loading word database: {e}")
        return

    # Start browser only once
    async with async_playwright() as p:
        try:
            browser_options = {}
            if LOCAL_CHROME_PATH:
                browser_options['executable_path'] = LOCAL_CHROME_PATH
                
            browser = await p.chromium.launch(
                headless=False,
                args=['--no-sandbox', '--disable-setuid-sandbox'],
                **browser_options
            )
            logger.info(f"Browser launched successfully")
        except Exception as e:
            logger.error(f"Error launching browser: {e}")
            return
        
        # Create a new browser context (like an incognito window)
        context = await browser.new_context()
        
        # Open game page
        try:
            page = await context.new_page()
            await page.goto(WORDLE_URL, wait_until='domcontentloaded')
            logger.info(f"Opened game page: {WORDLE_URL}")
            
            # Wait for page to be fully loaded
            await asyncio.sleep(5)  # Adjust as needed, per network speed
            logger.info(f"Page loaded, starting continuous solving")
            
            # Main game loop
            while True:
                solver = WordleSolver(words)  # Create new solver for each game
                
                try:
                    for attempt in range(MAX_ATTEMPTS):
                        # Get next guess
                        guess = solver.suggest_next_word()
                        if not guess:
                            logger.error(f"No valid guesses available")
                            break
                            
                        logger.info(f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: {guess}")
                        
                        # Type the guess and submit
                        await page.keyboard.type(guess)
                        await page.keyboard.press('Enter')
                        await page.wait_for_timeout(2000)
            
                        # Get the latest row with our guess
                        rows = await page.query_selector_all('.Row-locked-in')
                        if not rows:
                            logger.error(f"Could not find game row")
                            continue
                        current_row = rows[-1]
                        
                        # Analyze the result
                        letters = await current_row.query_selector_all('.Row-letter')
                        feedback = []
                        
                        for idx, letter_div in enumerate(letters):
                            classes = await letter_div.get_attribute('class')
                            letter_content = await letter_div.inner_text()
                            letter = ''.join(char for char in letter_content if char.isalpha()).strip().lower()
                            
                            if not letter:
                                logger.error(f"Could not extract letter from element with classes: {classes}")
                                continue
                            
                            if 'letter-correct' in classes:
                                feedback.append((letter, LetterState.CORRECT))
                            elif 'letter-elsewhere' in classes:
                                feedback.append((letter, LetterState.ELSEWHERE))
                            else:  # 'letter-absent'
                                feedback.append((letter, LetterState.ABSENT))
                        
                        # Update solver state with feedback
                        word_complete = solver.update_state(guess, feedback)
                        
                        if word_complete:
                            logger.success(f"Word solved in {attempt + 1} attempts!")
                            break
                            
                        if attempt == MAX_ATTEMPTS - 1:
                            logger.warning(f"Failed to solve the word")
                    
                    # After game ends, wait and restart
                    logger.info("Waiting to restart game...")
                    await page.wait_for_timeout(5000)  # Wait 5 seconds
                    
                    restart_button = await page.query_selector('.restart_btn')
                    if restart_button:
                        await restart_button.click()
                        await page.wait_for_timeout(1000)  # Wait 1 second after clicking
                        logger.info("Game restarted")
                    else:
                        logger.error("Could not find restart button")
                        raise Exception("Restart button not found")
                        
                except Exception as e:
                    logger.error(f"Error during game: {e}")
                    await browser.close()
                    return
                    
        except Exception as e:
            logger.error(f"Error opening game page: {e}")
        finally:
            await browser.close()
            logger.info(f"Browser closed")

async def main():
    try:
        await run_wordle_solver()
    except KeyboardInterrupt:
        logger.warning(f"Game interrupted by user")
    except Exception as e:
        logger.error(f"Error running main: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
