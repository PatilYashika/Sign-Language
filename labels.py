# List of all supported ISL signs, phrases, alphabet, numbers and words grouped by category

CATEGORIES = {
    "Numbers (1-9)": [
        "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ],
    "Alphabet (A-Z)": [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    ],
    "Animals & Wildlife": [
        "BEAR", "CROCODILE", "DEER", "ELEPHANT", "GIRAFFE",
        "LION", "MONKEY", "PEACOCK", "PIGEON", "SPARROW",
        "TIGER", "TURTLE", "VOLCANO"
    ],
    "Food, Vegetables & Cooking": [
        "BRINJAL", "CABBAGE", "CARROT", "CAULIFLOWER", "CHILLI",
        "COOK", "CUCUMBER", "DRINK", "LEMON", "ONION",
        "POUR", "RADISH", "TEA", "VEGETABLES"
    ],
    "Everyday Actions & Objects": [
        "BREAK", "BUDGET", "BUSY", "CLEAN", "CLOSE",
        "COME", "CRY", "EXAM", "FEDUP", "FEVER",
        "GIVE", "HUG", "INJURY", "INTERVIEW", "JUMP",
        "KARNATAKA", "KEY", "KNIFE", "MAN", "MATHS",
        "MAYBE", "STILL", "SWITCH", "TEMPLE", "UMBRELLA",
        "UNCLE", "WIFE", "WRITER", "WRONG"
    ],
    "Basic Greetings & Polite Exchanges": [
        "GOOD_MORNING",
        "GOOD_AFTERNOON",
        "GOOD_NIGHT",
        "HELLO",
        "THANK_YOU",
        "WHAT_IS_YOUR_NAME",
        "YES",
        "NO",
        "PLEASE",
        "NICE_TO_MEET_YOU",
        "SEE_YOU_LATER",
        "HAVE_A_GOOD_DAY",
        "SORRY",
        "EXCUSE_ME",
        "YOU_ARE_WELCOME",
        "MY_NAME_IS"
    ],
    "Daily Communication & Agreement": [
        "I_UNDERSTAND",
        "I_DO_NOT_UNDERSTAND",
        "CAN_YOU_REPEAT",
        "SPEAK_SLOWLY",
        "I_AGREE",
        "I_DO_NOT_KNOW",
        "OF_COURSE",
        "NEVER_MIND",
        "WAIT_A_MINUTE"
    ],
    "Needs, Actions & Requests": [
        "I_AM_HUNGRY",
        "I_AM_THIRSTY",
        "I_WANT_WATER",
        "I_WANT_FOOD",
        "I_AM_TIRED",
        "I_NEED_TO_GO",
        "CAN_I_HELP_YOU",
        "PLEASE_COME_HERE",
        "SIT_DOWN",
        "STAND_UP"
    ],
    "Common Questions & Navigation": [
        "WHERE_IS_THE_RESTROOM",
        "WHERE_IS_THE_HOSPITAL",
        "HOW_MUCH_DOES_IT_COST",
        "WHAT_ARE_YOU_DOING",
        "WHO_IS_THAT",
        "WHY_NOT",
        "WHICH_ONE",
        "WHEN_DO_WE_LEAVE",
        "DO_YOU_KNOW_SIGN_LANGUAGE",
        "WHERE_DO_YOU_LIVE"
    ],
    "Health, Safety & Emergency": [
        "CALL_AN_AMBULANCE",
        "CALL_THE_POLICE",
        "I_FEEL_SICK",
        "I_HAVE_PAIN",
        "I_AM_LOST",
        "IT_IS_AN_EMERGENCY",
        "DO_NOT_TOUCH",
        "BE_CAREFUL",
        "I_AM_FINE",
        "CALL_MY_FAMILY"
    ]
}

# Flat list of all labels
ALL_LABELS = []
for cat_labels in CATEGORIES.values():
    for label in cat_labels:
        if label not in ALL_LABELS:
            ALL_LABELS.append(label)

# 2 hands * 21 landmarks * 3 (x, y, z) = 126 features per frame
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS = 3
TOTAL_FEATURES = NUM_HANDS * LANDMARKS_PER_HAND * COORDS  # 126
SEQUENCE_LENGTH = 30  # 30 frames (~1 second at 30 fps)
