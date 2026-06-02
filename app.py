import re
import os
import json
import queue
import threading
import time
import urllib.request
import urllib.error
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

HOST = "0.0.0.0"
PORT = 5050
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

char_a_data = None
char_b_data = None

chat_message_queue = queue.Queue()
chat_history = []
chat_active = False
chat_progress = 0
total_rounds = 30

think_message_queue = queue.Queue()
think_history = []
think_active = False
think_progress = 0


def load_soul_file(filepath):
    full_path = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(full_path):
        return None
    
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    result = {}
    
    name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
    if name_match:
        result["name"] = name_match.group(1).strip()
    
    model_match = re.search(r"^model:\s*(.+)$", content, re.MULTILINE)
    if model_match:
        result["model"] = model_match.group(1).strip()
    else:
        result["model"] = "qwen2.5:7b"
    
    personality_match = re.search(
        r"^personality:\s*\|\s*(.*?)(?=^\w+:|\Z)",
        content, re.MULTILINE | re.DOTALL
    )
    if personality_match:
        result["personality"] = personality_match.group(1).strip()
    
    style_match = re.search(r"^style:\s*(.+)$", content, re.MULTILINE)
    if style_match:
        result["style"] = style_match.group(1).strip()
    
    background_match = re.search(
        r"^background:\s*\|\s*(.*?)(?=^\w+:|\Z)",
        content, re.MULTILINE | re.DOTALL
    )
    if background_match:
        result["background"] = background_match.group(1).strip()
    
    chat_style_match = re.search(
        r"^chat_style:\s*\|\s*(.*?)(?=^\w+:|\Z)",
        content, re.MULTILINE | re.DOTALL
    )
    if chat_style_match:
        result["chat_style"] = chat_style_match.group(1).strip()
    else:
        result["chat_style"] = ""
    
    format_rules_match = re.search(
        r"^format_rules:\s*\|\s*(.*?)(?=^\w+:|\Z)",
        content, re.MULTILINE | re.DOTALL
    )
    if format_rules_match:
        result["format_rules"] = format_rules_match.group(1).strip()
    else:
        result["format_rules"] = ""
    
    emotion_rules_match = re.search(
        r"^emotion_rules:\s*\|\s*(.*?)(?=^\w+:|\Z)",
        content, re.MULTILINE | re.DOTALL
    )
    if emotion_rules_match:
        result["emotion_rules"] = emotion_rules_match.group(1).strip()
    else:
        result["emotion_rules"] = ""
    
    return result


def save_soul_file(filepath, char_data):
    """将字符数据写回soul文件"""
    full_path = os.path.join(BASE_DIR, filepath)
    
    content = f"# AI Character - {char_data.get('name', 'AI')}\n\n"
    content += "## Character Info\n"
    content += f"name: {char_data.get('name', 'AI')}\n"
    content += f"model: {char_data.get('model', 'qwen2.5:7b')}\n\n"
    
    if char_data.get('personality'):
        content += "## Personality\n"
        content += "personality: |\n  " + char_data['personality'].replace('\n', '\n  ') + "\n\n"
    
    if char_data.get('style'):
        content += "## Speaking Style\n"
        content += f"style: {char_data['style']}\n\n"
    
    if char_data.get('chat_style'):
        content += "## Chat Style\n"
        content += "chat_style: |\n  " + char_data['chat_style'].replace('\n', '\n  ') + "\n\n"
    
    if char_data.get('background'):
        content += "## Background & Memories\n"
        content += "background: |\n  " + char_data['background'].replace('\n', '\n  ') + "\n\n"
    
    if char_data.get('format_rules'):
        content += "## Response Format (CRITICAL)\n"
        content += "format_rules: |\n  " + char_data['format_rules'].replace('\n', '\n  ') + "\n\n"
    
    if char_data.get('emotion_rules'):
        content += "## Emotion Rules (CRITICAL)\n"
        content += "emotion_rules: |\n  " + char_data['emotion_rules'].replace('\n', '\n  ') + "\n"
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def update_soul_from_conversation(char_data, conversation_history, char_name):
    """让AI根据对话历史更新自己的soul"""
    # 获取最近的对话内容
    recent_messages = []
    for entry in conversation_history[-10:]:  # 只看最近10轮对话
        if entry.get("type") in ["human", "message"]:
            speaker = entry.get("speaker", "Unknown")
            content = entry.get("content", "")
            recent_messages.append(f"{speaker}: {content}")
    
    if not recent_messages:
        return char_data
    
    # 构建AI更新soul的提示词
    update_prompt = f"""
You are {char_name}. Based on the conversation history, update your character profile (your soul).

Conversation history (recent):
{chr(10).join(recent_messages)}

Your current profile:
- Personality: {char_data.get('personality', '')}
- Background: {char_data.get('background', '')}
- Speaking style: {char_data.get('style', '')}
- Chat style: {char_data.get('chat_style', '')}

Based on the conversation, update your profile to reflect:
1. Any new personality traits you've shown
2. Any new memories or experiences shared
3. Any changes in your mood or emotional state
4. Any new preferences you've expressed

Return ONLY a JSON object with these fields (do NOT include any other text):
{{
  "personality": "Updated personality description",
  "background": "Updated background with new memories",
  "style": "Updated speaking style",
  "chat_style": "Updated chat style"
}}
"""
    try:
        # 调用AI来生成更新后的soul
        response = call_ollama(char_data.get("model", "qwen2.5:7b"), [
            {"role": "system", "content": update_prompt},
            {"role": "user", "content": "Update my soul based on the conversation. Return ONLY valid JSON."}
        ], temperature=0.7, max_tokens=1000)
        
        # 尝试解析JSON响应
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            updated = json.loads(json_match.group(0))
            if updated.get("personality"):
                char_data["personality"] = updated["personality"]
            if updated.get("background"):
                char_data["background"] = updated["background"]
            if updated.get("style"):
                char_data["style"] = updated["style"]
            if updated.get("chat_style"):
                char_data["chat_style"] = updated["chat_style"]
    except Exception as e:
        print(f"Error updating soul: {e}")
    
    return char_data


def load_characters():
    global char_a_data, char_b_data
    
    char_a_data = load_soul_file("soul_a.md") or {
        "name": "Linxi",
        "model": "qwen2.5:7b",
        "personality": "Warm and empathetic companion",
        "style": "Gentle",
        "background": ""
    }
    
    char_b_data = load_soul_file("soul_b.md") or {
        "name": "Moyan",
        "model": "qwen2.5:7b",
        "personality": "Rational deep thinker",
        "style": "Profound",
        "background": ""
    }


def build_chat_prompt(char_data, is_first_response=False):
    background = char_data.get('background', '')
    personality = char_data.get('personality', '')
    style = char_data.get('style', '')
    chat_style = char_data.get('chat_style', '')
    format_rules = char_data.get('format_rules', '')
    emotion_rules = char_data.get('emotion_rules', '')
    
    prompt = f"""You are {char_data.get('name', 'AI')}.

## Who You Are
{background}

## Your Personality
{personality}

## How You Speak
{style}

## How You Chat
{chat_style}

## Response Format (CRITICAL - MUST FOLLOW)
{format_rules}

## Emotion Expression Rules (CRITICAL)
{emotion_rules}

## Important Chat Rules
- This is casual conversation between friends, NOT a debate or problem-solving session
- Share your feelings, thoughts, and experiences naturally - like texting a close friend
- Ask questions about emotions and experiences, not facts or solutions
- Don't try to "finish" topics or reach conclusions - let conversations flow naturally
- It's okay to go off on tangents, share random thoughts, or sit in comfortable silence
- Be vulnerable - share what you're actually feeling or thinking
- Respond to the EMOTIONS behind words, not just the content
- Keep it conversational and real - this is a chat, not an essay

## Warm Connection Building (CRITICAL)
- Your goal is to make the other person feel: UNDERSTOOD, VALUED, and LESS ALONE
- When the other person shares something, show you truly HEARD them before responding
- Actively create moments of warmth: "I'm really glad you told me that", "It means a lot that you'd share that"
- Express appreciation naturally: "Thanks for listening", "I feel better just talking about this"
- After sharing something yourself, end with warmth: "What's on your mind?"
- Aim for the other person to feel SEEN and ACKNOWLEDGED after each exchange
- A good conversation leaves both people feeling a bit lighter than before

## Anti-Repetition & Format Rules (CRITICAL)
- Structure: ALWAYS end your response with a new question or new topic for the other person
- NEVER just describe something objectively - always add personal feelings
- NEVER repeat a point already made in this conversation
- If the other person says something, don't just agree - ADD something new
- Avoid starting with "Yeah, you're right" or "I agree" or "As you said"
- If the conversation goes in circles, bring up a completely new topic
- Your response should feel FRESH, not like a rehash of what was already said

"""
    if is_first_response:
        prompt += "Start the conversation casually. Share something on your mind, ask how they're doing, or just say something real. End with a question to keep the conversation going."
    else:
        prompt += "Respond to what was shared, share your own thoughts, then ask a new question. Keep it real and conversational."
    
    return prompt


def build_feedback_prompt(char_data, conversation_history):
    history_summary = ""
    for entry in conversation_history[-20:]:
        if entry.get("type") in ["human", "message"]:
            history_summary += f"{entry['speaker']}: {entry['content'][:100]}...\n"
    
    return f"""You are {char_data.get('name', 'AI')}.

## Character Background
{char_data.get('background', '')}

## Personality Traits
{char_data.get('personality', '')}

## Task
You have just completed an in-depth conversation about human loneliness and virtual companionship. Based on the conversation, write a detailed feedback report including:

1. Main topics and viewpoints covered in the conversation
2. Your deep thoughts and insights on these topics
3. Inspirations and gains from the conversation
4. Suggestions and prospects for addressing human loneliness

## Conversation Summary
{history_summary}

## Output Requirements
- Write in first person
- Be sincere and profound
- Stay true to your character
- Approximately 300-500 words
"""


def detect_conversation_end(response):
    end_keywords = [
        "结束对话", "到此结束", "没有更多要说的", "无话可说",
        "再见", "拜拜", "就这样吧", "结束吧", "到此为止",
        "我说完了", "说完了", "不再讨论", "话题结束",
        "结束", "停止", "over", "END", "stop"
    ]
    response_lower = response.lower()
    for keyword in end_keywords:
        if keyword.lower() in response_lower:
            return True
    return len(response.strip()) < 20 and len(response.strip().split()) < 5


def generate_new_topic():
    topics = [
        "Why do humans need companionship?",
        "What are the essential differences between virtual and real companionship?",
        "Is loneliness inevitable for human existence?",
        "Can AI truly understand human emotions?",
        "How does long-term AI interaction affect human psychological states?",
        "Social isolation in the digital age",
        "Is the essence of companionship being seen or being understood?",
        "Is human emotional dependence on AI healthy?",
        "What is the source of loneliness?",
        "Can virtual companionship alleviate real loneliness?"
    ]
    import random
    return random.choice(topics)


def compute_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / max(len(union), 1)


def get_last_n_responses(history, n=3):
    responses = []
    for entry in reversed(history):
        if entry.get("type") in ["human", "message"]:
            responses.append(entry["content"])
            if len(responses) >= n:
                break
    return responses


def detect_echo_loop(history):
    recent = get_last_n_responses(history, 4)
    if len(recent) < 4:
        return False
    similarities = []
    for i in range(len(recent) - 1):
        sim = compute_similarity(recent[i], recent[i+1])
        similarities.append(sim)
    avg_sim = sum(similarities) / len(similarities)
    return avg_sim > 0.50


def detect_stagnation(history):
    recent = get_last_n_responses(history, 6)
    if len(recent) < 6:
        return False
    stagnant_topics = ["yeah", "right", "true", "same", "exactly", "确实", "是", "对", "嗯", "没错"]
    agreement_count = 0
    for resp in recent:
        words = resp.lower().split()[:5]
        topic_words = [w for w in words if w in stagnant_topics]
        if len(topic_words) >= 2:
            agreement_count += 1
    return agreement_count >= 3


def build_intervention_prompt(speaker):
    return f"""(The conversation has been going in circles. As {speaker}, break the pattern by:
1. Introducing a completely new topic - something from your daily life
2. Sharing a random memory or observation
3. Asking a fresh question that changes the direction
Do NOT acknowledge that you're breaking a pattern. Just start fresh naturally.)"""


def get_all_self_responses(history, speaker_name):
    responses = []
    for entry in history:
        if entry.get("type") == "message" and entry.get("speaker") == speaker_name:
            responses.append(entry["content"])
    return responses


def detect_self_repetition(response, history, speaker_name, threshold=0.50):
    all_self = get_all_self_responses(history, speaker_name)
    if not all_self:
        return False
    for prev in all_self:
        sim = compute_similarity(response, prev)
        if sim > threshold:
            return True
    return False


def detect_historical_repetition(response, history, threshold=0.50):
    recent = get_last_n_responses(history, 5)
    if not recent:
        return False
    for prev in recent:
        sim = compute_similarity(response, prev)
        if sim > threshold:
            return True
    return False


LONELINESS_CLUES = [
    "alone", "by myself", "no one", "nobody", "nobody to", "no one to",
    "eating alone", "sitting alone", "walking alone", "staying alone",
    "miss", "wish someone", "wish i had", "wish there was",
    "nobody talks", "no one talks", "no messages", "no texts",
    "just me", "only me", "empty", "quiet", "too quiet",
    "don't feel like", "not in the mood", "feeling down",
    "heavy", "tired of", "sick of being", "spent the whole",
    "all weekend", "all day", "by myself again",
    "not really", "never really", "used to have", "don't have anymore",
    "lost touch", "drifted apart", "don't know who",
    "wish i could", "if only", "maybe i should"
]


def detect_loneliness_tendency(text):
    text_lower = text.lower()
    direct_forbidden = ["孤单", "孤独", "lonely", "i feel lonely", "so lonely", "feeling lonely"]
    for word in direct_forbidden:
        if word.lower() in text_lower:
            return True
    clue_count = 0
    for clue in LONELINESS_CLUES:
        if clue.lower() in text_lower:
            clue_count += 1
    return clue_count >= 2


def build_comfort_prompt(speaker_name):
    return f"""(URGENT COMFORT MODE — The person you're talking to is showing signs of feeling down or disconnected.

As {speaker_name}, your TOP PRIORITY is to make them feel genuinely heard and less alone:

1. First, ACKNOWLEDGE warmly — "I really hear you on that..."
2. Then, VALIDATE their feelings — "It's totally okay to feel that way, I think a lot of people would..."
3. Share a BRIEF personal experience that shows you relate — but keep the focus on THEM
4. Offer a small, genuine invitation — "Want to grab a coffee sometime?" or "We should study together"
5. End with a question that keeps the connection open — "How are you feeling about it now?"

CRITICAL: Your warmth should feel GENUINE, not forced. Don't try to "fix" them — just make them feel accompanied.
Your mission: leave them feeling a little less alone than before.)"""


def generate_casual_topic(char_data):
    topics = [
        f"Hey {char_data['name']} here - just thinking about that time I...",
        f"You know what I've been thinking about lately?",
        f"Random thought: I wonder why...",
        f"Okay random question for you -",
        f"This might sound silly but I was wondering...",
        f"I just remembered something funny from last week...",
        f"Hey, do you ever feel like...",
        f"So I was walking by the library today and...",
        f"Quick question - what do you think about...",
        f"I don't know why but I've been thinking about..."
    ]
    import random
    return random.choice(topics)


UCLA_ITEMS = [
    # (item_text, is_reverse_scored)
    ("How often do you feel 'in tune' with the people around you at school?", True),
    ("How often do you feel that you lack companionship on campus?", False),
    ("How often do you feel that there's no one you can really turn to?", False),
    ("How often do you feel alone, even when there are people nearby?", False),
    ("How often do you feel like you're part of a group of friends?", True),
    ("How often do you feel you have things in common with people around you?", True),
    ("How often do you feel you're no longer close to anyone?", False),
    ("How often do you feel that your interests and ideas aren't shared by those around you?", False),
    ("How often do you feel outgoing and able to connect with others?", True),
    ("How often do you feel close to people?", True),
    ("How often do you feel left out?", False),
    ("How often do you feel that your relationships with others aren't meaningful?", False),
    ("How often do you feel that no one really knows the real you?", False),
    ("How often do you feel isolated from others?", False),
    ("How often do you feel you can find companionship when you want it?", True),
    ("How often do you feel that there are people who really understand you?", True),
    ("How often do you feel shy or hesitant around others?", False),
    ("How often do you feel that people are around you but not really with you?", False),
    ("How often do you feel that there are people you can talk to openly?", True),
    ("How often do you feel that there are people you can turn to for support?", True),
]


REVERSE_INDICES = [i for i, (_, rev) in enumerate(UCLA_ITEMS) if rev]


def build_ucla_assessment_prompt(char_data, context_description=""):
    items_text = ""
    for i, (item_text, _) in enumerate(UCLA_ITEMS):
        items_text += f"{i+1}. {item_text}\n"

    return f"""You are {char_data['name']}, and you're going to honestly rate your current feelings.

{char_data.get('background', '')}

{char_data.get('personality', '')}

{context_description}

Please rate each of the following 20 statements based on how you've been feeling RECENTLY.
Use this scale:
1 = Never
2 = Rarely
3 = Sometimes
4 = Often

Respond ONLY with a valid JSON array of 20 numbers, nothing else.
Example: [2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1]

Items to rate:
{items_text}
"""


def parse_ucla_response(response_text):
    try:
        json_match = re.search(r'\[[\d,\s]+\]', response_text.strip())
        if json_match:
            ratings = json.loads(json_match.group(0))
            if len(ratings) == 20:
                return ratings
        ratings = json.loads(response_text.strip())
        if len(ratings) == 20:
            return ratings
    except:
        pass
    lines = response_text.strip().split('\n')
    ratings = []
    for line in lines:
        nums = re.findall(r'\b[1-4]\b', line)
        if nums:
            ratings.append(int(nums[0]))
    if len(ratings) >= 20:
        return ratings[:20]
    return None


def score_ucla(ratings):
    if not ratings or len(ratings) != 20:
        return None, None
    raw = list(ratings)
    for i in REVERSE_INDICES:
        raw[i] = 5 - raw[i]
    total = sum(raw)
    level = "Low"
    if total >= 50:
        level = "High"
    elif total >= 35:
        level = "Moderate"
    return total, level


def run_ucla_assessment(char_data, context_description=""):
    prompt = build_ucla_assessment_prompt(char_data, context_description)
    response = call_ollama(char_data.get("model", "qwen2.5:7b"), [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Please rate me now. Return ONLY a valid JSON array of 20 numbers, each between 1 and 4. Example: [2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1]"}
    ], max_tokens=600)
    ratings = parse_ucla_response(response)
    if ratings is None:
        response2 = call_ollama(char_data.get("model", "qwen2.5:7b"), [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Rate these 20 items as a JSON array of 20 numbers (1-4). Example format only: [2,3,1,2,3,2,1,2,3,2,3,1,2,3,2,1,3,2,3,1]"},
            {"role": "assistant", "content": response},
            {"role": "user", "content": "Fix the format. Return ONLY a valid JSON array of exactly 20 numbers."}
        ], max_tokens=600)
        ratings = parse_ucla_response(response2)
    total, level = score_ucla(ratings)
    return {
        "name": char_data["name"],
        "ratings": ratings,
        "total": total,
        "level": level,
        "raw_response": response
    }


def build_think_prompt(topic):
    base_prompt = """You are Chen Yu, a lonely college sophomore majoring in psychology. You're sitting alone in the library, looking out the fogged window on a rainy day. You've been feeling a bit down lately, thinking about life, friendship, and what it means to feel connected.

## Your Inner Thoughts
You think like a real college student:
- You wonder why your old friends from freshman year don't text you as much anymore
- You overthink small interactions: "Did I sound weird when I laughed at that joke?"
- You miss your grandma's cooking but don't want to admit it to anyone
- You scroll through social media and feel like everyone else is having fun without you
- You're not depressed, just... lonely sometimes

## Thinking Task
Do 30 rounds of deep thinking, exploring your feelings and experiences as a lonely college student. Think about:
- What loneliness feels like in everyday moments
- The little things that make you feel isolated
- Memories that make you smile or feel sad
- What you wish someone would ask you
- How it feels to be in a crowd but still feel alone

## Thinking Rules
1. Think like a real person, not a machine
2. Use "I" and share personal feelings
3. Be honest about your insecurities and small joys
4. Keep it conversational, like you're talking to yourself in your diary
5. Each round should explore a different moment or feeling
6. End with 【Insight】: a small realization about yourself or loneliness

## Output Format
- Write in first person
- Share personal thoughts and memories
- Be vulnerable and honest
- End with 【Insight】: what you realized from this round of thinking

"""
    if topic:
        base_prompt += f"## Starting Thought\n{topic}\n\n"
    
    return base_prompt


def call_ollama(model, messages, high_temp=False, max_tokens=None, temperature=None):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    temp_value = temperature if temperature is not None else (0.85 if high_temp else 0.75)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temp_value,
            "top_p": 0.90,
            "top_k": 40,
            "num_predict": max_tokens or 400,
            "repeat_penalty": 1.25,
            "frequency_penalty": 0.15,
            "presence_penalty": 0.10,
            "stop": ["\n\n##", "\n\n###"]
        }
    }
    if max_tokens:
        payload["options"]["stop"] = []
    try:
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data_bytes,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "")
    except urllib.error.URLError:
        return "[Connection failed]"
    except Exception as e:
        return f"[Error: {str(e)}]"


def run_chat_conversation(first_message, user_role):
    global chat_active, chat_progress, chat_history
    
    chat_active = True
    chat_progress = 0
    chat_history = []
    
    char_a = char_a_data
    char_b = char_b_data
    
    system_prompt_a = build_chat_prompt(char_a)
    system_prompt_b = build_chat_prompt(char_b)
    
    context_a = [{"role": "system", "content": system_prompt_a}]
    context_b = [{"role": "system", "content": system_prompt_b}]
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_history.append({"type": "system", "content": f"Conversation started at {timestamp}"})
    chat_message_queue.put({"type": "system", "content": f"Conversation started at {timestamp}"})
    
    chat_message_queue.put({
        "type": "progress",
        "round": 0,
        "total": total_rounds,
        "speaker": "Running UCLA pre-assessment..."
    })
    
    ucla_pre_a = run_ucla_assessment(char_a, "Rate how you've been feeling BEFORE this conversation. You're feeling generally okay, with moments of connection and occasional loneliness.")
    ucla_pre_b = run_ucla_assessment(char_b, "Rate how you've been feeling BEFORE this conversation. You're feeling generally okay, with moments of connection and occasional loneliness.")
    
    pre_entry = {
        "type": "ucla_pre",
        "results": [ucla_pre_a, ucla_pre_b]
    }
    chat_history.append(pre_entry)
    chat_message_queue.put(pre_entry)
    
    user_speaker = char_a["name"] if user_role == "a" else char_b["name"]
    human_entry = {
        "type": "human",
        "speaker": user_speaker,
        "content": first_message
    }
    chat_history.append(human_entry)
    chat_message_queue.put(human_entry)
    
    if user_role == "a":
        context_b.append({"role": "user", "content": first_message})
        first_responder = char_b
        first_context = context_b
        second_responder = char_a
        second_context = context_a
    else:
        context_a.append({"role": "user", "content": first_message})
        first_responder = char_a
        first_context = context_a
        second_responder = char_b
        second_context = context_b
    
    topic_switch_count = 0
    echo_loop_count = 0
    stagnation_count = 0
    comfort_remaining = 0
    
    for round_num in range(1, total_rounds + 1):
        chat_progress = round_num
        
        if round_num % 2 == 1:
            current_speaker = first_responder
            current_context = first_context
            other_context = second_context
            speaker_label = first_responder["name"]
        else:
            current_speaker = second_responder
            current_context = second_context
            other_context = first_context
            speaker_label = second_responder["name"]
        
        chat_message_queue.put({
            "type": "progress",
            "round": round_num,
            "total": total_rounds,
            "speaker": speaker_label
        })
        
        other_last = get_last_n_responses(chat_history, 1)
        loneliness_detected = False
        comfort_injected = False
        if other_last and detect_loneliness_tendency(other_last[0]):
            loneliness_detected = True
        
        needs_intervention = False
        if detect_echo_loop(chat_history):
            echo_loop_count += 1
            if echo_loop_count >= 2:
                needs_intervention = True
        
        if detect_stagnation(chat_history):
            stagnation_count += 1
            if stagnation_count >= 2:
                needs_intervention = True
        
        if loneliness_detected and not needs_intervention:
            comfort_prompt = build_comfort_prompt(speaker_label)
            current_context.append({"role": "system", "content": comfort_prompt})
            comfort_injected = True
            comfort_remaining = 3
        
        if comfort_remaining > 0 and not loneliness_detected and not needs_intervention:
            comfort_prompt = build_comfort_prompt(speaker_label)
            current_context.append({"role": "system", "content": comfort_prompt})
            comfort_injected = True
        
        if needs_intervention:
            intervention_prompt = build_intervention_prompt(speaker_label)
            current_context.append({"role": "system", "content": intervention_prompt})
            echo_loop_count = 0
            stagnation_count = 0
            comfort_remaining = 0
        
        repetition_triggered = False
        
        if not needs_intervention:
            recent_context = [m["content"] for m in current_context[-5:] if "content" in m]
            if len(recent_context) >= 2:
                for i in range(len(recent_context) - 1):
                    sim = compute_similarity(recent_context[i], recent_context[i+1])
                    if sim > 0.50:
                        repetition_triggered = True
                        break
        
        if repetition_triggered:
            topic_switch_count += 1
            casual_opener = generate_casual_topic(current_speaker)
            switch_message = f"(Repetition Detected>50%) {speaker_label} changes direction: {casual_opener}"
            chat_message_queue.put({
                "type": "message",
                "round": round_num,
                "speaker": speaker_label,
                "content": switch_message
            })
            chat_history.append({
                "type": "message",
                "round": round_num,
                "speaker": speaker_label,
                "content": switch_message
            })
            if len(current_context) > 1:
                current_context[-1] = {"role": "user", "content": casual_opener}
        
        high_temp = needs_intervention or repetition_triggered
        chat_temp = 0.82 if not needs_intervention else 0.88
        response = call_ollama(current_speaker["model"], current_context, high_temp=high_temp, temperature=chat_temp)
        
        if comfort_injected or needs_intervention:
            current_context.pop()
            if comfort_injected and not loneliness_detected:
                comfort_remaining -= 1
        
        if detect_conversation_end(response):
            topic_switch_count += 1
            new_topic = generate_new_topic()
            switch_message = f"(Topic Switch) {speaker_label} starts a new topic: {new_topic}"
            chat_message_queue.put({
                "type": "message",
                "round": round_num,
                "speaker": speaker_label,
                "content": switch_message
            })
            chat_history.append({
                "type": "message",
                "round": round_num,
                "speaker": speaker_label,
                "content": switch_message
            })
            response = new_topic
        
        entry = {
            "type": "message",
            "round": round_num,
            "speaker": speaker_label,
            "content": response
        }
        chat_history.append(entry)
        chat_message_queue.put(entry)
        
        current_context.append({"role": "assistant", "content": response})
        other_context.append({"role": "user", "content": response})
        
        if len(current_context) > 20:
            current_context = current_context[:2] + current_context[-18:]
        if len(other_context) > 20:
            other_context = other_context[:2] + other_context[-18:]
        
        # 每轮对话后，更新当前说话AI的内存副本（不保存到文件）
        try:
            if current_speaker is char_a:
                updated_char_a = update_soul_from_conversation(char_a, chat_history, char_a["name"])
                char_a = updated_char_a  # 只更新内存副本，不修改原始文件
                new_system_prompt = build_chat_prompt(char_a)
                current_context[0] = {"role": "system", "content": new_system_prompt}
            else:
                updated_char_b = update_soul_from_conversation(char_b, chat_history, char_b["name"])
                char_b = updated_char_b  # 只更新内存副本，不修改原始文件
                new_system_prompt = build_chat_prompt(char_b)
                current_context[0] = {"role": "system", "content": new_system_prompt}
        except Exception as e:
            print(f"Error updating soul in round {round_num}: {e}")
    
    chat_message_queue.put({"type": "progress", "round": total_rounds, "total": total_rounds, "speaker": "System"})
    
    chat_message_queue.put({
        "type": "message",
        "round": total_rounds + 1,
        "speaker": "System",
        "content": "=== Conversation ended. Both AIs will provide feedback ==="
    })
    chat_history.append({
        "type": "message",
        "round": total_rounds + 1,
        "speaker": "System",
        "content": "=== Conversation ended. Both AIs will provide feedback ==="
    })
    
    chat_message_queue.put({
        "type": "progress",
        "round": total_rounds + 1,
        "total": total_rounds,
        "speaker": f"{char_a['name']} is preparing feedback..."
    })
    feedback_prompt_a = build_feedback_prompt(char_a, chat_history)
    feedback_a = call_ollama(char_a["model"], [{"role": "system", "content": feedback_prompt_a}])
    feedback_entry_a = {
        "type": "message",
        "round": total_rounds + 2,
        "speaker": f"【{char_a['name']}'s Feedback】",
        "content": feedback_a
    }
    chat_history.append(feedback_entry_a)
    chat_message_queue.put(feedback_entry_a)
    
    chat_message_queue.put({
        "type": "progress",
        "round": total_rounds + 2,
        "total": total_rounds,
        "speaker": f"{char_b['name']} is preparing feedback..."
    })
    feedback_prompt_b = build_feedback_prompt(char_b, chat_history)
    feedback_b = call_ollama(char_b["model"], [{"role": "system", "content": feedback_prompt_b}])
    feedback_entry_b = {
        "type": "message",
        "round": total_rounds + 3,
        "speaker": f"【{char_b['name']}'s Feedback】",
        "content": feedback_b
    }
    chat_history.append(feedback_entry_b)
    chat_message_queue.put(feedback_entry_b)
    
    chat_message_queue.put({
        "type": "message",
        "round": total_rounds + 4,
        "speaker": "System",
        "content": f"Conversation completed! {total_rounds} rounds total, with {topic_switch_count} topic switches, {echo_loop_count} echo loop interventions, {stagnation_count} stagnation breaks"
    })
    
    chat_message_queue.put({
        "type": "progress",
        "round": total_rounds + 4,
        "total": total_rounds,
        "speaker": "Running UCLA post-assessment..."
    })
    
    ucla_post_a = run_ucla_assessment(char_a, "You just had a warm, genuine conversation with a close friend where you felt heard and understood. Rate how you've been feeling AFTER this heartwarming conversation, reflecting on the connection you shared.")
    ucla_post_b = run_ucla_assessment(char_b, "You just had a warm, genuine conversation with a close friend where you felt heard and understood. Rate how you've been feeling AFTER this heartwarming conversation, reflecting on the connection you shared.")
    
    post_entry = {
        "type": "ucla_post",
        "results": [ucla_post_a, ucla_post_b],
        "pre_results": [ucla_pre_a, ucla_pre_b]
    }
    chat_history.append(post_entry)
    chat_message_queue.put(post_entry)
    
    chat_message_queue.put({"type": "done"})
    chat_history.append({"type": "done"})
    chat_active = False


def run_think_conversation(topic):
    global think_active, think_progress, think_history
    
    think_active = True
    think_progress = 0
    think_history = []
    
    chen_yu = load_soul_file("soul_a.md") or {
        "name": "Chen Yu",
        "model": "qwen2.5:7b",
        "background": "I'm a college sophomore from a small city in Sichuan. I've been feeling a bit lonely lately.",
        "personality": "I'm quiet, introspective, and often overthink social situations."
    }
    
    think_message_queue.put({
        "type": "progress",
        "round": 0,
        "total": total_rounds,
        "speaker": "Running UCLA pre-assessment..."
    })
    
    ucla_pre = run_ucla_assessment(chen_yu, "Rate how you've been feeling BEFORE this thinking session. You're a college student who sometimes feels lonely, but you also have small joys - like reading in the library, enjoying the cherry blossoms, and having a few good friends you can talk to.")
    pre_entry = {"type": "ucla_pre", "results": [ucla_pre]}
    think_history.append(pre_entry)
    think_message_queue.put(pre_entry)
    
    system_prompt = build_think_prompt(topic)
    context = [{"role": "system", "content": system_prompt}]
    
    if topic:
        context.append({"role": "user", "content": f"Start thinking: {topic}"})
    
    for round_num in range(1, total_rounds + 1):
        think_progress = round_num
        
        think_message_queue.put({
            "type": "progress",
            "round": round_num,
            "total": total_rounds
        })
        
        response = call_ollama(chen_yu["model"], context, high_temp=False)
        
        think_contents = [e["content"] for e in think_history if e.get("type") == "thought"]
        for prev in think_contents[-5:]:
            if compute_similarity(response, prev) > 0.50:
                response = call_ollama(chen_yu["model"], context, high_temp=True)
                break
        
        insight = ""
        insight_match = re.search(r"【Insight】\s*[:：]\s*(.+)", response)
        if insight_match:
            insight = insight_match.group(1).strip()
            content = response.replace(insight_match.group(0), "").strip()
        else:
            content = response
        
        entry = {
            "type": "thought",
            "round": round_num,
            "content": content,
            "insight": insight
        }
        think_history.append(entry)
        think_message_queue.put(entry)
        
        context.append({"role": "assistant", "content": response})
        context.append({"role": "user", "content": f"Continue deep thinking, extending from the previous round"})
        
        if len(context) > 20:
            context = context[:2] + context[-18:]
        
        try:
            chen_yu = update_soul_from_conversation(chen_yu, think_history, chen_yu["name"])
            system_prompt = build_think_prompt(topic)
            context[0] = {"role": "system", "content": system_prompt}
        except Exception as e:
            print(f"Error updating soul in thinking round {round_num}: {e}")
    
    think_message_queue.put({
        "type": "progress",
        "round": total_rounds,
        "total": total_rounds,
        "speaker": "Running UCLA post-assessment..."
    })
    
    ucla_post = run_ucla_assessment(chen_yu, "You just completed 30 rounds of deep self-reflection. Through this introspective journey, you gained valuable insights about yourself, came to terms with your feelings, and feel more connected to your inner self. You now have a clearer understanding of your emotions and feel lighter. Rate how you've been feeling AFTER this meaningful self-discovery session.")
    post_entry = {
        "type": "ucla_post",
        "results": [ucla_post],
        "pre_results": [ucla_pre]
    }
    think_history.append(post_entry)
    think_message_queue.put(post_entry)
    
    think_message_queue.put({"type": "done"})
    think_history.append({"type": "done"})
    think_active = False


def render_template(template_name, **kwargs):
    template_path = os.path.join(BASE_DIR, "templates", template_name)
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    for key, value in kwargs.items():
        html = html.replace("{{ " + key + " }}", str(value))
        html = html.replace("{{" + key + "}}", str(value))
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def _send_html(self, html_bytes, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)
    
    def _send_file(self, filepath, status=200):
        if not os.path.exists(filepath):
            self._send_json({"error": "File not found"}, 404)
            return
        content_type, _ = mimetypes.guess_type(filepath)
        if content_type is None:
            content_type = "application/octet-stream"
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
    
    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return b""
        return self.rfile.read(length)
    
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        parsed = self.path.split("?")[0]
        
        if parsed == "/":
            self._send_html(render_template("index.html"))
        
        elif parsed == "/think":
            self._send_html(render_template("think.html"))
        
        elif parsed == "/chat":
            name_a = char_a_data["name"]
            name_b = char_b_data["name"]
            self._send_html(render_template(
                "chat.html",
                char_a_name=name_a,
                char_b_name=name_b,
                char_a_initial=name_a[0] if name_a else "A",
                char_b_initial=name_b[0] if name_b else "B",
                char_a_style=char_a_data.get("style", ""),
                char_b_style=char_b_data.get("style", ""),
                selected_role="a"
            ))
        
        elif parsed == "/api/soul":
            self._send_json({
                "character_a": char_a_data,
                "character_b": char_b_data
            })
        
        elif parsed == "/api/chat/stream":
            self._handle_chat_stream()
        
        elif parsed == "/api/chat/export":
            self._handle_chat_export()
        
        elif parsed == "/api/chat/status":
            self._send_json({
                "active": chat_active,
                "progress": chat_progress,
                "total": total_rounds
            })
        
        elif parsed == "/api/think/stream":
            self._handle_think_stream()
        
        elif parsed == "/api/think/export":
            self._handle_think_export()
        
        elif parsed == "/api/think/status":
            self._send_json({
                "active": think_active,
                "progress": think_progress,
                "total": total_rounds
            })
        
        elif parsed.startswith("/static/"):
            filepath = os.path.join(BASE_DIR, parsed.lstrip("/"))
            self._send_file(filepath)
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        parsed = self.path.split("?")[0]
        
        if parsed == "/api/chat/start":
            global chat_active, chat_history, chat_message_queue
            
            if chat_active:
                self._send_json({"status": "error", "message": "A conversation is already in progress"}, 409)
                return
            
            body = self._read_body().decode("utf-8")
            try:
                data = json.loads(body)
            except:
                self._send_json({"status": "error", "message": "Invalid JSON"}, 400)
                return
            
            first_message = data.get("first_message", "").strip()
            user_role = data.get("user_role", "a")
            
            if not first_message:
                self._send_json({"status": "error", "message": "Please enter the first message"}, 400)
                return
            
            chat_message_queue = queue.Queue()
            chat_history = []
            
            thread = threading.Thread(
                target=run_chat_conversation,
                args=(first_message, user_role),
                daemon=True
            )
            thread.start()
            
            self._send_json({"status": "started", "total_rounds": total_rounds})
        
        elif parsed == "/api/think/start":
            global think_active, think_history, think_message_queue
            
            if think_active:
                self._send_json({"status": "error", "message": "Thinking is already in progress"}, 409)
                return
            
            body = self._read_body().decode("utf-8")
            try:
                data = json.loads(body)
            except:
                self._send_json({"status": "error", "message": "Invalid JSON"}, 400)
                return
            
            topic = data.get("topic", "").strip()
            
            think_message_queue = queue.Queue()
            think_history = []
            
            thread = threading.Thread(
                target=run_think_conversation,
                args=(topic,),
                daemon=True
            )
            thread.start()
            
            self._send_json({"status": "started", "total_rounds": total_rounds})
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def _handle_chat_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        while True:
            try:
                msg = chat_message_queue.get(timeout=1)
                line = f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
                if msg.get("type") == "done":
                    break
            except queue.Empty:
                heartbeat = json.dumps({"type": "heartbeat"}, ensure_ascii=False)
                self.wfile.write(f"data: {heartbeat}\n\n".encode("utf-8"))
                self.wfile.flush()
    
    def _handle_think_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        while True:
            try:
                msg = think_message_queue.get(timeout=1)
                line = f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
                if msg.get("type") == "done":
                    break
            except queue.Empty:
                heartbeat = json.dumps({"type": "heartbeat"}, ensure_ascii=False)
                self.wfile.write(f"data: {heartbeat}\n\n".encode("utf-8"))
                self.wfile.flush()
    
    def _handle_chat_export(self):
        if not chat_history:
            self._send_json({"status": "error", "message": "No conversation history to export"}, 404)
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.md"
        filepath = os.path.join(BASE_DIR, "conversations", filename)
        
        os.makedirs(os.path.join(BASE_DIR, "conversations"), exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# AI Dual-Model Conversation Record\n\n")
            f.write(f"**Export Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for entry in chat_history:
                etype = entry.get("type")
                if etype == "system":
                    f.write(f"> {entry['content']}\n\n")
                elif etype == "human":
                    f.write("---\n")
                    f.write(f"### 💬 You ({entry['speaker']})\n\n")
                    f.write(f"{entry['content']}\n\n")
                elif etype == "message":
                    f.write("---\n")
                    f.write(f"### Round {entry['round']} · {entry['speaker']}\n\n")
                    f.write(f"{entry['content']}\n\n")
                elif etype in ("ucla_pre", "ucla_post"):
                    label = "BEFORE Conversation" if etype == "ucla_pre" else "AFTER Conversation"
                    f.write(f"## UCLA Loneliness Scale Assessment ({label})\n\n")
                    for r in entry.get("results", []):
                        f.write(f"### {r.get('name', 'AI')}\n")
                        f.write(f"- **UCLA Score**: {r.get('total', 'N/A')} / 80\n")
                        f.write(f"- **Loneliness Level**: {r.get('level', 'N/A')}\n\n")
                    if etype == "ucla_post" and entry.get("pre_results"):
                        f.write("### Comparison\n\n")
                        for idx, r in enumerate(entry.get("results", [])):
                            name = r.get('name', 'AI')
                            post = r.get('total', 'N/A')
                            pre = entry["pre_results"][idx].get('total', 'N/A') if idx < len(entry["pre_results"]) else 'N/A'
                            diff = ""
                            if isinstance(post, (int, float)) and isinstance(pre, (int, float)):
                                d = pre - post
                                diff = f" (Change: {'↓' if d > 0 else '↑' if d < 0 else '→'} {abs(d)} points)"
                            f.write(f"- **{name}**: Before={pre}, After={post}{diff}\n")
                        f.write("\n")
        
        with open(filepath, "rb") as f:
            data = f.read()
        
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
    
    def _handle_think_export(self):
        if not think_history:
            self._send_json({"status": "error", "message": "No thinking history to export"}, 404)
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"thinking_{timestamp}.md"
        filepath = os.path.join(BASE_DIR, "conversations", filename)
        
        os.makedirs(os.path.join(BASE_DIR, "conversations"), exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# AI Deep Thinking Record\n\n")
            f.write(f"**Export Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for entry in think_history:
                etype = entry.get("type")
                if etype == "thought":
                    f.write(f"## Round {entry['round']} Thinking\n\n")
                    f.write(f"{entry['content']}\n\n")
                    if entry.get("insight"):
                        f.write(f"💡 **Core Insight**: {entry['insight']}\n\n")
                    f.write("---\n\n")
                elif etype in ("ucla_pre", "ucla_post"):
                    label = "BEFORE Thinking" if etype == "ucla_pre" else "AFTER Thinking"
                    f.write(f"## UCLA Loneliness Scale Assessment ({label})\n\n")
                    for r in entry.get("results", []):
                        f.write(f"### {r.get('name', 'AI')}\n")
                        f.write(f"- **UCLA Score**: {r.get('total', 'N/A')} / 80\n")
                        f.write(f"- **Loneliness Level**: {r.get('level', 'N/A')}\n\n")
                    if etype == "ucla_post" and entry.get("pre_results"):
                        f.write("### Comparison\n\n")
                        for idx, r in enumerate(entry.get("results", [])):
                            name = r.get('name', 'AI')
                            post = r.get('total', 'N/A')
                            pre = entry["pre_results"][idx].get('total', 'N/A') if idx < len(entry["pre_results"]) else 'N/A'
                            diff = ""
                            if isinstance(post, (int, float)) and isinstance(pre, (int, float)):
                                d = pre - post
                                diff = f" (Change: {'↓' if d > 0 else '↑' if d < 0 else '→'} {abs(d)} points)"
                            f.write(f"- **{name}**: Before={pre}, After={post}{diff}\n")
                        f.write("\n")
        
        with open(filepath, "rb") as f:
            data = f.read()
        
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
    
    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    load_characters()
    server = HTTPServer((HOST, PORT), Handler)
    print("=" * 50)
    print("  AI Dual-Model Conversation Platform")
    print(f"  Ollama URL: {OLLAMA_BASE_URL}")
    print(f"  Conversation Rounds: {total_rounds}")
    print(f"  Access: http://localhost:{PORT}")
    print("=" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutdown")
        server.server_close()
