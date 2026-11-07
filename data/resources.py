# data/resources.py
# Building blocks of the dataset, written by hand.
# The impolite sentences are rude only by their tone (orders, impatience), never with insults.
# Templates marked True are "held out": only used in test_unseen.
# slots
NAMES = [
    "Sarah", "Ahmed", "Maria", "Yuki", "Daniel", "Fatima", "Lucas", "Amina", "Chen", "Olivia",
    "Karim", "Sofia", "Mateo", "Priya", "Ethan", "Leila", "Noah", "Ines", "Omar", "Hannah",
    "Ravi", "Elena", "Youssef", "Grace", "Tom", "Nadia", "Kenji", "Laura", "Samuel", "Aisha",
    "Victor", "Mei", "Julia", "Hugo", "Zainab", "Paul", "Anna", "Diego", "Salma", "Adam",
    "Lina", "Marc", "Rania", "Felix", "Chloe", "Ali", "Emma", "Rosa", "Ivan", "Nora",
]

DOCS = [
    "report", "presentation", "invoice", "contract", "spreadsheet", "slides", "budget",
    "proposal", "summary", "agenda", "schedule", "draft", "timeline", "brochure", "newsletter",
    "survey results", "meeting notes", "project plan", "price list", "user guide", "cover letter",
    "monthly report", "sales figures", "lesson plan", "research paper", "floor plan",
]

EVENTS = [
    "meeting", "team lunch", "workshop", "conference call", "training session", "birthday party",
    "staff meeting", "project review", "client visit", "board meeting", "book club",
    "football match", "piano lesson", "job interview", "group study session", "farewell dinner",
    "product launch", "parents' evening", "yoga class", "planning session",
]

PLACES = [
    "office", "kitchen", "meeting room", "library", "lab", "classroom", "hallway", "lobby",
    "break room", "reception", "warehouse", "garden", "parking lot", "studio",
]

ITEMS = [
    "laptop", "charger", "umbrella", "stapler", "headphones", "keys", "camera", "projector",
    "bike", "notebook", "calculator", "tablet", "printer cable", "ladder", "tent",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLOCK = ["8 am", "9 am", "9:30", "10 am", "11 am", "noon", "1 pm", "2 pm", "3:30", "4 pm",
         "5 pm", "6 pm", "7 pm"]
CITIES = ["Paris", "Tunis", "Berlin", "Madrid", "Rome", "Lisbon", "Cairo", "Montreal",
          "Tokyo", "Dublin", "Vienna", "Casablanca"]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]
PRICES = ["ten dollars", "twenty euros", "fifteen dollars", "forty euros", "five dollars",
          "thirty euros", "a hundred dollars", "twelve euros"]
TOPICS = ["the budget", "the website", "the new clients", "the training program",
          "the office move", "the hiring process", "the annual survey", "the social media account",
          "the warehouse inventory", "the summer interns"]

# (impolite time, polite time): "now" becomes "as soon as possible"
TIMES = [
    ("today", "today"), ("by Friday", "by Friday"), ("before noon", "before noon"),
    ("by the end of the day", "by the end of the day"), ("this afternoon", "this afternoon"),
    ("tomorrow morning", "tomorrow morning"), ("before the meeting", "before the meeting"),
    ("by 5 pm", "by 5 pm"), ("this week", "this week"), ("tonight", "tonight"),
    ("before Monday", "before Monday"), ("by tomorrow", "by tomorrow"),
    ("now", "as soon as possible"), ("right now", "as soon as possible"),
    ("immediately", "as soon as possible"), ("", ""),
]

# actions
# (base, -ing, past participle, complement)
ACTIONS = [
    ("send", "sending", "sent", "me the {doc}"),
    ("send", "sending", "sent", "the {doc} to {name}"),
    ("send", "sending", "sent", "the updated {doc} to the whole team"),
    ("check", "checking", "checked", "the numbers in the {doc}"),
    ("check", "checking", "checked", "the {doc} for mistakes"),
    ("review", "reviewing", "reviewed", "the {doc}"),
    ("review", "reviewing", "reviewed", "{name}'s {doc}"),
    ("update", "updating", "updated", "the {doc}"),
    ("update", "updating", "updated", "the shared calendar"),
    ("update", "updating", "updated", "the website with the new prices"),
    ("fix", "fixing", "fixed", "the printer in the {place}"),
    ("fix", "fixing", "fixed", "the broken link on the website"),
    ("fix", "fixing", "fixed", "the typo in the {doc}"),
    ("finish", "finishing", "finished", "the {doc}"),
    ("finish", "finishing", "finished", "your part of the {doc}"),
    ("prepare", "preparing", "prepared", "the {doc} for the {event}"),
    ("prepare", "preparing", "prepared", "the room for the {event}"),
    ("share", "sharing", "shared", "the {doc} with {name}"),
    ("share", "sharing", "shared", "your notes from the {event}"),
    ("call", "calling", "called", "{name} back"),
    ("call", "calling", "called", "the supplier about the delivery"),
    ("call", "calling", "called", "the client about the {doc}"),
    ("clean", "cleaning", "cleaned", "the {place}"),
    ("clean", "cleaning", "cleaned", "the whiteboard after the {event}"),
    ("move", "moving", "moved", "your {item} out of the {place}"),
    ("move", "moving", "moved", "the {event} to {day}"),
    ("move", "moving", "moved", "the boxes out of the {place}"),
    ("reply", "replying", "replied", "to my email about the {doc}"),
    ("reply", "replying", "replied", "to {name}'s message"),
    ("book", "booking", "booked", "a room for the {event}"),
    ("book", "booking", "booked", "the train tickets to {city}"),
    ("sign", "signing", "signed", "the {doc}"),
    ("sign", "signing", "signed", "the attendance sheet for the {event}"),
    ("print", "printing", "printed", "two copies of the {doc}"),
    ("print", "printing", "printed", "the {doc} for the {event}"),
    ("return", "returning", "returned", "my {item}"),
    ("return", "returning", "returned", "the {item} to the {place}"),
    ("bring", "bringing", "brought", "your {item} to the {event}"),
    ("bring", "bringing", "brought", "the {doc} to the {event}"),
    ("explain", "explaining", "explained", "the {doc} to {name}"),
    ("explain", "explaining", "explained", "the new procedure to the team"),
    ("confirm", "confirming", "confirmed", "the date of the {event}"),
    ("confirm", "confirming", "confirmed", "your attendance at the {event}"),
    ("translate", "translating", "translated", "the {doc} into Spanish"),
    ("upload", "uploading", "uploaded", "the {doc} to the shared folder"),
    ("organize", "organizing", "organized", "the files in the {place}"),
    ("submit", "submitting", "submitted", "your timesheet"),
    ("submit", "submitting", "submitted", "the {doc} to {name}"),
    ("pay", "paying", "paid", "the invoice from the supplier"),
    ("lock", "locking", "locked", "the {place} when you leave"),
    ("remind", "reminding", "reminded", "{name} about the {event}"),
    ("schedule", "scheduling", "scheduled", "a call with {name}"),
    ("water", "watering", "watered", "the plants in the {place}"),
]

# things somebody refuses to do: (base, -ing)
ACTIVITIES = [
    ("come to the {event}", "coming to the {event}"),
    ("join the {event}", "joining the {event}"),
    ("cover your shift on {day}", "covering your shift on {day}"),
    ("lend you my {item}", "lending you my {item}"),
    ("help you move on {day}", "helping you move on {day}"),
    ("drive you to the airport", "driving you to the airport"),
    ("work this weekend", "working this weekend"),
    ("stay late on {day}", "staying late on {day}"),
    ("give a talk at the {event}", "giving a talk at the {event}"),
    ("take on another project", "taking on another project"),
    ("organize the {event}", "organizing the {event}"),
    ("rewrite the {doc} for you", "rewriting the {doc} for you"),
    ("look after your plants next week", "looking after your plants next week"),
    ("pick {name} up from the station", "picking {name} up from the station"),
    ("present the {doc} at the {event}", "presenting the {doc} at the {event}"),
    ("travel to {city} next month", "traveling to {city} next month"),
    ("answer emails during my holiday", "answering emails during my holiday"),
    ("swap my day off with you", "swapping my day off with you"),
]

# when (only added if the activity has no time)
REFUSAL_WHEN = ["", "tomorrow", "this weekend", "tonight", "next week", "on {day}", "this time",
                "on {day} morning", "after work"]

# where / when (only added if the behaviour has none)
STOP_CONTEXT = ["", "in the {place}", "during the {event}", "every morning", "when I am on the phone",
                "during lunch", "in front of the clients", "in the middle of the day", "at the {event}"]

# habits somebody is asked to stop
BEHAVIOURS = [
    "talking during the presentation", "using my {item} without asking",
    "parking in my spot", "playing music so loud", "interrupting me",
    "leaving your dishes in the sink", "eating at your desk during calls",
    "calling me after 9 pm", "changing the {doc} without telling me",
    "sending me messages on the weekend", "taking my chair", "leaving the {place} door open",
    "booking the meeting room every day", "forwarding my emails to {name}",
    "printing personal documents at work", "tapping your pen", "moving my files",
    "arriving late to the {event}", "making phone calls in the {place}",
    "reading my screen over my shoulder", "eating my food from the fridge",
    "putting your bags on the table", "turning off the heating", "sitting on my desk",
    "whistling", "leaving your coffee cups on my desk", "using the speakerphone",
    "borrowing my {item} without returning it", "skipping the {event}", "clicking your pen",
    "talking over {name}", "opening the windows", "humming", "asking me for my password",
    "rearranging the chairs", "watching videos without headphones", "slamming the door",
    "copying me on every email", "rescheduling the {event} at the last minute",
    "posting about work on social media", "leaving your {item} in the {place}",
]

# pieces of work that get criticised, by type
THINGS = {
    "text": ["report", "essay", "email", "summary", "proposal", "cover letter", "newsletter",
             "speech", "survey", "recipe", "article", "contract"],
    "visual": ["logo", "poster", "design", "slides", "chart", "map", "brochure", "menu", "flyer"],
    "software": ["code", "website", "app", "spreadsheet"],
    "plan": ["plan", "budget", "schedule", "timeline"],
    "media": ["video", "playlist", "podcast"],
}

# what the piece of work is for
THING_CONTEXT = ["", "for the {event}", "about {topic}", "for {name}", "you sent this morning",
                 "for the client", "for {day}", "on {topic}", "for the {event} on {day}"]

# (harsh version, soft version, types of work)
ISSUES = [
    ("way too long", "is a little long", {"text", "media", "plan"}),
    ("impossible to read", "is a bit hard to read", {"text", "visual"}),
    ("full of mistakes", "has a few mistakes", {"text", "software", "plan"}),
    ("a total mess", "is a bit disorganized", {"text", "visual", "plan"}),
    ("way too slow", "is a little slow", {"software", "media"}),
    ("so confusing", "is a bit confusing", {"text", "visual", "software", "plan"}),
    ("really boring", "is not very engaging yet", {"text", "media", "visual"}),
    ("way too expensive", "is a bit expensive", {"plan"}),
    ("far too short", "is a little short", {"text", "media"}),
    ("completely out of date", "is slightly out of date", {"text", "visual", "software", "plan"}),
    ("way too busy", "looks a little crowded", {"visual"}),
    ("so badly organized", "is not very well organized", {"text", "plan", "visual"}),
    ("too vague", "is a bit vague", {"text", "plan"}),
    ("full of typos", "has a few typos", {"text", "visual"}),
    ("way too dark", "is a little dark", {"visual", "media"}),
    ("way too complicated", "is a bit complicated", {"text", "plan", "software"}),
    ("unrealistic", "seems a bit ambitious", {"plan"}),
    ("way too loud", "is a little loud", {"media"}),
]

# services and their problems: (problem, tense)
# "perfect" -> "for three days", "past" -> "yesterday", "present" -> nothing
P = {
    "late": ("is late again", "present"), "not_arrived": ("has still not arrived", "perfect"),
    "charged_twice": ("was charged twice", "past"), "wrong_address": ("was sent to the wrong address", "past"),
    "cancelled": ("was cancelled without notice", "past"), "not_working": ("has not been working", "perfect"),
    "cutting_out": ("keeps cutting out", "present"), "slow": ("has been very slow", "perfect"),
    "broken": ("has been broken", "perfect"), "not_fixed": ("has not been fixed", "perfect"),
    "wrong": ("is wrong again", "present"), "blocked": ("has been blocked", "perfect"),
    "not_finished": ("is still not finished", "present"), "disappeared": ("disappeared from the system", "past"),
    "changed": ("was changed without notice", "past"), "damaged": ("arrived damaged", "past"),
    "noisy": ("is very noisy", "present"), "jamming": ("keeps jamming", "present"),
    "delayed": ("has been delayed", "perfect"),
}
SERVICES = {
    "order": [P[k] for k in ("late", "not_arrived", "charged_twice", "wrong_address", "cancelled", "delayed")],
    "delivery": [P[k] for k in ("late", "not_arrived", "wrong_address", "delayed")],
    "package": [P[k] for k in ("not_arrived", "wrong_address", "damaged", "delayed")],
    "refund": [P[k] for k in ("not_arrived", "late", "delayed")],
    "internet connection": [P[k] for k in ("not_working", "cutting_out", "slow")],
    "heating": [P[k] for k in ("not_working", "broken", "not_fixed")],
    "washing machine": [P[k] for k in ("broken", "not_working", "not_fixed", "noisy")],
    "phone bill": [P[k] for k in ("charged_twice", "wrong")],
    "subscription": [P[k] for k in ("charged_twice", "cancelled", "blocked")],
    "online account": [P[k] for k in ("blocked", "not_working")],
    "laptop repair": [P[k] for k in ("late", "not_finished", "delayed")],
    "hotel booking": [P[k] for k in ("cancelled", "disappeared", "changed")],
    "flight booking": [P[k] for k in ("cancelled", "changed", "delayed")],
    "parking permit": [P[k] for k in ("not_arrived", "blocked")],
    "air conditioning": [P[k] for k in ("not_working", "broken", "noisy")],
    "elevator": [P[k] for k in ("not_working", "broken", "not_fixed")],
    "water heater": [P[k] for k in ("not_working", "broken")],
    "library card": [P[k] for k in ("blocked", "not_working")],
    "printer": [P[k] for k in ("not_working", "broken", "jamming")],
    "train": [P[k] for k in ("late", "cancelled", "delayed")],
}
DURATIONS = {
    "perfect": ["for three days", "since {day}", "for a week", "since this morning", "for two weeks",
                "since yesterday", "for an hour", "since last night", "for ten days", "for a month",
                "since the weekend", "for two hours", ""],
    "past": ["yesterday", "this morning", "on {day}", "last week", "again", "two days ago", ""],
    "present": [""],
}
# services with a number ("my order number 4521")
NUMBERED_SERVICES = {"order", "package", "refund", "hotel booking", "flight booking", "delivery"}

# facts the other person got wrong
FACTS = [
    "the meeting starts at {clock}", "the deadline is on {day}", "the {doc} is due on {day}",
    "the office is on the {ordinal} floor", "the ticket costs {price}",
    "the train leaves at {clock}", "the exam is on {day}", "the store closes at {clock}",
    "{name} is in charge of {topic}", "the conference is in {city}",
    "the {event} is on {day}", "the {event} starts at {clock}",
    "the {doc} was sent to {name}", "the {place} is closed on {day}",
    "the bus stops in front of the {place}", "{name} is responsible for {topic}",
    "the package weighs two kilos", "the {item} belongs to {name}",
]

# (direct question, embedded question)
QUESTIONS = [
    ("where is the {doc}", "where the {doc} is"),
    ("when does the {event} start", "when the {event} starts"),
    ("who is responsible for {topic}", "who is responsible for {topic}"),
    ("how much does the {item} cost", "how much the {item} costs"),
    ("what time does the {place} close", "what time the {place} closes"),
    ("why was the {event} cancelled", "why the {event} was cancelled"),
    ("where did you put my {item}", "where you put my {item}"),
    ("when will the {doc} be ready", "when the {doc} will be ready"),
    ("who took the {item} from the {place}", "who took the {item} from the {place}"),
    ("what is the password for the wifi", "what the password for the wifi is"),
    ("how long will the {event} last", "how long the {event} will last"),
    ("which room is the {event} in", "which room the {event} is in"),
    ("when is {name} coming back", "when {name} is coming back"),
    ("where is the nearest {place}", "where the nearest {place} is"),
    ("when does the train to {city} leave", "when the train to {city} leaves"),
    ("is the {place} open on {day}", "whether the {place} is open on {day}"),
    ("where is {name}'s office", "where {name}'s office is"),
    ("what did {name} say about the {doc}", "what {name} said about the {doc}"),
    ("how do I get to the {place}", "how I can get to the {place}"),
    ("can I use the {item} on {day}", "whether I can use the {item} on {day}"),
    ("is the {event} still on {day}", "whether the {event} is still on {day}"),
    ("where can I find a {item}", "where I can find a {item}"),
    ("what is {name}'s phone number", "what {name}'s phone number is"),
    ("who has the key to the {place}", "who has the key to the {place}"),
    ("when will my {item} be repaired", "when my {item} will be repaired"),
    ("who is going to the {event} with {name}", "who is going to the {event} with {name}"),
    ("what time does the {event} end on {day}", "what time the {event} ends on {day}"),
    ("why is the {place} closed on {day}", "why the {place} is closed on {day}"),
    ("who sent the {doc} to {name}", "who sent the {doc} to {name}"),
    ("when did {name} send the {doc}", "when {name} sent the {doc}"),
    ("what is the deadline for the {doc}", "what the deadline for the {doc} is"),
    ("how do I open the {doc}", "how I can open the {doc}"),
]

# changes to an event
CHANGES = [
    "is cancelled", "has been moved to {day}", "is postponed until {day}",
    "now starts at {clock}", "has been moved to the {place}", "will be online this time",
    "is shorter this week", "will end at {clock}", "has been moved to next week",
    "has been moved to {day} at {clock}", "will take place in the {place}",
    "now starts at {clock} on {day}", "will be led by {name}", "has been moved to {city}",
    "is postponed until further notice", "will start one hour later on {day}",
    "has been moved from the {place} to the {place2}", "is now on {day} instead",
    "is full, so registration is closed", "will be recorded for {name}",
]

# templates
# each template: (id, text, held_out)
TEMPLATES = {
    "request": {
        "impolite": [
            ("req_01", "{Base} {comp} {time}.", False),
            ("req_02", "Just {base} {comp} {time}.", False),
            ("req_03", "{Base} {comp} {time}, and stop wasting my time.", False),
            ("req_04", "I told you to {base} {comp}. Do it {time}.", False),
            ("req_05", "How many times do I have to ask? {Base} {comp} {time}.", False),
            ("req_06", "{Base} {comp} {time}. I am not asking twice.", False),
            ("req_07", "Go {base} {comp} {time}.", False),
            ("req_08", "You need to {base} {comp} {time}. No excuses.", False),
            ("req_09", "{Base} {comp} {time} already!", False),
            ("req_10", "Hurry up and {base} {comp} {time}.", True),
            ("req_11", "Is it so hard to {base} {comp} {time}? Do it.", True),
            ("req_12", "I expect you to {base} {comp} {time}, period.", True),
        ],
        "polite": [
            "Could you please {base} {comp} {ptime}?",
            "Would you mind {ing} {comp} {ptime}?",
            "When you have a moment, could you {base} {comp} {ptime}?",
            "I would really appreciate it if you could {base} {comp} {ptime}.",
            "Would it be possible for you to {base} {comp} {ptime}?",
            "Could I ask you to {base} {comp} {ptime}? Thank you!",
        ],
    },
    "reminder": {
        "impolite": [
            ("rem_01", "You still haven't {pp} {comp}. Do it {time}.", False),
            ("rem_02", "Why haven't you {pp} {comp} yet?", False),
            ("rem_03", "I'm still waiting. {Base} {comp} {time}.", False),
            ("rem_04", "It's been days and you still haven't {pp} {comp}.", False),
            ("rem_05", "For the last time, {base} {comp} {time}.", False),
            ("rem_06", "Seriously, you were supposed to {base} {comp}. {Base} it {time}.", False),
            ("rem_07", "Don't forget again to {base} {comp} {time}.", False),
            ("rem_08", "Have you {pp} {comp} or not?", False),
            ("rem_09", "I'm tired of reminding you to {base} {comp}.", False),
            ("rem_10", "Still no news. Did you even {base} {comp}?", True),
            ("rem_11", "Do I have to remind you every day to {base} {comp}?", True),
            ("rem_12", "You keep forgetting to {base} {comp}. Do it {time}.", True),
        ],
        "polite": [
            "Just a friendly reminder to {base} {comp} {ptime}.",
            "I wanted to follow up: could you {base} {comp} {ptime}?",
            "Sorry to bother you again, but could you please {base} {comp} {ptime}?",
            "Could you let me know when you have {pp} {comp}? Thank you.",
            "Whenever you get a chance, could you {base} {comp} {ptime}?",
            "I'm just checking in: have you had a chance to {base} {comp}?",
        ],
    },
    "refusal": {
        "impolite": [
            ("ref_01", "No. I'm not {aing}.", False),
            ("ref_02", "Forget it, I won't {abase}.", False),
            ("ref_03", "I'm not going to {abase}, stop asking.", False),
            ("ref_04", "Not happening. I won't {abase}.", False),
            ("ref_05", "Ask someone else, I'm not {aing}.", False),
            ("ref_06", "Why would I {abase}? No.", False),
            ("ref_07", "No way I'm {aing}.", False),
            ("ref_08", "Don't even ask me to {abase}.", False),
            ("ref_09", "I already said no to {aing}.", False),
            ("ref_10", "Absolutely not, I won't {abase}.", True),
            ("ref_11", "Find someone else to {abase}, I'm done.", True),
            ("ref_12", "You can't expect me to {abase}. No.", True),
        ],
        "polite": [
            "I'm sorry, but I won't be able to {abase}.",
            "Unfortunately, I can't {abase}, but thank you for asking.",
            "Thank you for thinking of me, but I'm unable to {abase}.",
            "I'd love to help, but I can't {abase} this time.",
            "I'm afraid I won't be able to {abase}. I hope you understand.",
            "Thanks for asking, but I have to say no to {aing}.",
        ],
    },
    "criticism": {
        "impolite": [
            ("cri_01", "Your {thing} is {harsh}.", False),
            ("cri_02", "This {thing} is {harsh}. Redo it.", False),
            ("cri_03", "Did you even check the {thing}? It's {harsh}.", False),
            ("cri_04", "Honestly, the {thing} is {harsh}.", False),
            ("cri_05", "The {thing} is {harsh}, as usual.", False),
            ("cri_06", "I can't accept this {thing}. It's {harsh}.", False),
            ("cri_07", "What is this {thing}? It's {harsh}.", False),
            ("cri_08", "Your {thing} is {harsh}. Fix it.", False),
            ("cri_09", "Nobody will like this {thing}, it's {harsh}.", False),
            ("cri_10", "Seriously, the {thing} is {harsh}.", True),
            ("cri_11", "Look at this {thing}. It's {harsh}.", True),
            ("cri_12", "Again? The {thing} is {harsh}.", True),
        ],
        "polite": [
            "Thank you for the {thing}. I think it {soft}, though.",
            "The {thing} is a good start, but I feel it {soft}.",
            "I noticed the {thing} {soft}. Could we work on that together?",
            "Thanks for your work on the {thing}. In my opinion it {soft}.",
            "I appreciate the effort on the {thing}, but it {soft}.",
            "Could we take another look at the {thing}? It {soft}.",
        ],
    },
    "complaint": {
        "impolite": [
            ("com_01", "My {service} {problem} {dur}. This is ridiculous.", False),
            ("com_02", "Unbelievable. The {service} {problem} {dur}.", False),
            ("com_03", "I'm fed up, my {service} {problem} {dur}.", False),
            ("com_04", "Fix it now, my {service} {problem} {dur}.", False),
            ("com_05", "What kind of service is this? My {service} {problem} {dur}.", False),
            ("com_06", "Again! The {service} {problem} {dur}.", False),
            ("com_07", "Do your job. My {service} {problem} {dur}.", False),
            ("com_08", "This is the worst service ever, the {service} {problem} {dur}.", False),
            ("com_09", "I want answers now. My {service} {problem} {dur}.", False),
            ("com_10", "Typical. The {service} {problem} {dur}.", True),
            ("com_11", "How is this acceptable? The {service} {problem} {dur}.", True),
            ("com_12", "I've had enough, my {service} {problem} {dur}.", True),
        ],
        "polite": [
            "Hello, I wanted to let you know that my {service} {problem} {dur}. Could you please look into it?",
            "I'm sorry to bother you, but my {service} {problem} {dur}. Could you help me?",
            "Excuse me, it seems that my {service} {problem} {dur}. Would you be able to check?",
            "Hi, I'm writing because my {service} {problem} {dur}. I would appreciate your help.",
            "Could you please help me? My {service} {problem} {dur}.",
            "I hope you can help: my {service} {problem} {dur}. Thank you in advance.",
        ],
    },
    "disagreement": {
        "impolite": [
            ("dis_01", "That's wrong. {Fact}.", False),
            ("dis_02", "No, you're wrong, {fact}.", False),
            ("dis_03", "Are you even listening? {Fact}.", False),
            ("dis_04", "Obviously {fact}. Get it right.", False),
            ("dis_05", "Wrong again, {fact}.", False),
            ("dis_06", "How do you not know that {fact}?", False),
            ("dis_07", "Stop guessing, {fact}.", False),
            ("dis_08", "Everyone knows that {fact}.", False),
            ("dis_09", "Check your facts. {Fact}.", False),
            ("dis_10", "Nope, {fact}. Pay attention.", True),
            ("dis_11", "That makes no sense, {fact}.", True),
            ("dis_12", "I already told you that {fact}.", True),
        ],
        "polite": [
            "I might be mistaken, but I believe {fact}.",
            "Actually, I think {fact}. Could you double-check?",
            "Just to clarify, I think {fact}.",
            "If I remember correctly, {fact}.",
            "Sorry, I think there may be a small mix-up: {fact}.",
            "I could be wrong, but as far as I know {fact}.",
        ],
    },
    "question": {
        "impolite": [
            ("que_01", "{Direct}? Answer me.", False),
            ("que_02", "So {direct}? I don't have all day.", False),
            ("que_03", "Just tell me {embedded}.", False),
            ("que_04", "{Direct}? Hurry up.", False),
            ("que_05", "Tell me {embedded}, now.", False),
            ("que_06", "I asked you twice, {direct}?", False),
            ("que_07", "Seriously, {direct}?", False),
            ("que_08", "Somebody tell me {embedded}.", False),
            ("que_09", "Come on, {direct}?", False),
            ("que_10", "Well? {Direct}?", True),
            ("que_11", "Why can't anyone tell me {embedded}?", True),
            ("que_12", "I need to know {embedded}. Now.", True),
        ],
        "polite": [
            "Could you tell me {embedded}, please?",
            "Excuse me, do you know {embedded}?",
            "Would you mind telling me {embedded}?",
            "I was wondering if you could tell me {embedded}.",
            "Sorry to bother you, but do you happen to know {embedded}?",
            "Could you please let me know {embedded}?",
        ],
    },
    "stop": {
        "impolite": [
            ("sto_01", "Stop {behaviour}.", False),
            ("sto_02", "Quit {behaviour}, seriously.", False),
            ("sto_03", "Can you stop {behaviour} already?", False),
            ("sto_04", "I've told you to stop {behaviour}.", False),
            ("sto_05", "Enough with {behaviour}.", False),
            ("sto_06", "Stop {behaviour}, it's annoying.", False),
            ("sto_07", "No more {behaviour}, understood?", False),
            ("sto_08", "Cut it out with {behaviour}.", False),
            ("sto_09", "Will you ever stop {behaviour}?", False),
            ("sto_10", "I'm sick of you {behaviour}.", True),
            ("sto_11", "Knock it off with {behaviour}.", True),
            ("sto_12", "How many times? Stop {behaviour}.", True),
        ],
        "polite": [
            "Would you mind not {behaviour}, please?",
            "Could you please avoid {behaviour}?",
            "I'd appreciate it if you could stop {behaviour}.",
            "Sorry to bring this up, but could you stop {behaviour}?",
            "Would it be possible to avoid {behaviour} from now on?",
            "If you don't mind, could you avoid {behaviour}? Thank you.",
        ],
    },
    "announcement": {
        "impolite": [
            ("ann_01", "The {event} {change}. Deal with it.", False),
            ("ann_02", "The {event} {change}. Don't complain.", False),
            ("ann_03", "The {event} {change}, not my problem.", False),
            ("ann_04", "The {event} {change}. Stop asking me about it.", False),
            ("ann_05", "Whatever, the {event} {change}.", False),
            ("ann_06", "Read your emails, the {event} {change}.", False),
            ("ann_07", "The {event} {change}. Get used to it.", False),
            ("ann_08", "For the last time, the {event} {change}.", False),
            ("ann_09", "The {event} {change}, end of discussion.", False),
            ("ann_10", "Like I said, the {event} {change}. Move on.", True),
            ("ann_11", "Nobody cares, the {event} {change}.", True),
            ("ann_12", "The {event} {change}. Take it or leave it.", True),
        ],
        "polite": [
            "Unfortunately, the {event} {change}. Sorry for the inconvenience.",
            "Please note that the {event} {change}. Thank you for your understanding.",
            "I wanted to let you know that the {event} {change}.",
            "Just a quick update: the {event} {change}. I apologize for any trouble.",
            "I'm sorry to say that the {event} {change}. Thanks for your patience.",
            "For your information, the {event} {change}. Please let me know if you have questions.",
        ],
    },
}

CATEGORY_DESCRIPTIONS = {
    "request": "asking someone to do something",
    "reminder": "following up on something not done yet",
    "refusal": "declining a request or an invitation",
    "criticism": "giving negative feedback on a piece of work",
    "complaint": "complaining about a service problem",
    "disagreement": "correcting a fact someone got wrong",
    "question": "asking for information",
    "stop": "asking someone to stop a behaviour",
    "announcement": "announcing a change to an event",
}

# words that must never appear in the dataset (checked on every sentence)
BANNED_WORDS = {
    "idiot", "idiots", "stupid", "dumb", "moron", "fool", "fools", "loser", "shut", "hell",
    "damn", "crap", "shit", "fuck", "fucking", "ass", "bitch", "bastard", "jerk", "pathetic",
    "useless", "retard", "retarded", "hate", "kill", "die", "ugly", "fat", "lazy", "incompetent",
    "garbage", "trash", "suck", "sucks", "clown", "freak", "disgusting", "worthless", "nonsense",
}


# copy of everything above = first version of the dataset (to regenerate data_v1/)
import copy as _copy
V1 = _copy.deepcopy({k: v for k, v in dict(globals()).items() if k.isupper() and not k.startswith("_")})
V1["ITEMS"].insert(V1["ITEMS"].index("calculator") + 1, "car")   # "car" was removed in v2


# VERSION 2
# The first version only talked about office life and the models failed on my own sentences,
# so I added home life, shopping, transport, school, more objects and verbs,
# and an optional reason sentence ("We are already late.") that must be kept.

NAMES += [
    "Bilal", "Clara", "Jonas", "Meriem", "Arthur", "Sana", "Lea", "Mehdi", "Ruth", "Tariq",
    "Camille", "Malik", "Eva", "Hamza", "Iris", "Jamal", "Keiko", "Luca", "Maya", "Nabil",
    "Oscar", "Petra", "Quentin", "Reem", "Simon", "Tara", "Umar", "Vera", "Walid", "Yara",
]
DOCS += [
    "quote", "receipt", "application form", "registration form", "timetable", "seating plan",
    "shopping list", "packing list", "guest list", "menu", "map", "photos", "minutes",
    "training manual", "press release", "job description", "insurance form", "lab report",
    "reading list", "exam schedule",
]
EVENTS += [
    "dinner party", "wedding", "school trip", "family lunch", "dentist appointment", "concert",
    "hiking trip", "barbecue", "open day", "graduation ceremony", "doctor's appointment",
    "team dinner", "cooking class", "swimming lesson", "charity run", "movie night",
]
PLACES += ["staff room", "storeroom", "cafeteria", "dining room", "common room", "print room"]
ITEMS += [
    "suitcase", "toolbox", "flashlight", "drill", "blender", "sleeping bag",
    "speaker", "phone charger", "rain jacket", "board game", "backpack", "hair dryer", "vacuum cleaner", "stroller", "skateboard", "guitar",
]
CITIES += ["London", "Oslo", "Prague", "Athens", "Istanbul", "Marrakesh", "Lyon", "Geneva",
           "Seoul", "Toronto", "Brussels", "Amsterdam", "Nairobi", "Lima"]
TOPICS += ["the school trip", "the holiday rota", "the new menu", "the parking rules",
           "the cleaning schedule", "the garden project", "the library renovation", "the charity sale"]

# everyday objects
OPENABLE = ["window", "door", "back door", "front door", "gate", "curtains", "blinds",
            "garage door", "kitchen window", "bathroom door", "balcony door", "car window",
            "cupboard", "drawer", "fridge", "shutters"]
DEVICES = ["heater", "TV", "radio", "fan", "oven", "computer", "alarm", "air conditioner",
           "coffee machine", "stove", "dishwasher", "washing machine", "projector", "microwave",
           "printer", "kettle", "dryer", "laptop"]
LOUD = ["music", "TV", "radio", "volume", "sound", "speaker", "video", "game"]
OBJ_S = ["umbrella", "ladder", "suitcase", "toolbox", "flashlight", "laptop", "charger", "camera",
         "notebook", "calculator", "stapler", "tablet", "jacket", "backpack", "map", "blanket",
         "lunch box", "water bottle", "phone", "wallet", "remote control", "hammer", "bucket",
         "basket", "mirror", "lamp", "pillow", "towel", "helmet", "bicycle pump"]
OBJ_P = ["keys", "chairs", "boxes", "bags", "books", "shoes", "plates", "cups", "glasses",
         "coats", "tools", "groceries", "papers", "folders", "bottles", "pens", "tickets",
         "receipts", "flowers", "plants", "dishes", "towels", "toys", "magazines", "batteries",
         "boots", "gloves", "scissors", "cables", "photos"]
FOOD = ["a coffee", "a cup of tea", "a glass of water", "a sandwich", "some cake", "some soup",
        "a slice of pizza", "some juice", "some bread", "some fruit", "a salad", "some cookies",
        "a bottle of water", "some biscuits", "a hot chocolate"]
FOOD_N = ["sandwich", "yogurt", "salad", "lunch", "chocolate", "cake", "pizza", "apple",
          "cookies", "soup", "cheese", "leftovers"]
SUPPLIES = ["printer paper", "milk", "coffee", "bread", "batteries", "ink cartridges", "stamps",
            "envelopes", "toilet paper", "dish soap", "light bulbs", "water bottles", "sugar",
            "pens", "notebooks", "cleaning products", "paper towels", "rice", "eggs", "tea bags"]
ROOMS = ["kitchen", "bathroom", "living room", "bedroom", "hallway", "garage", "attic",
         "basement", "balcony", "dining room", "office", "classroom", "staff room", "lab",
         "storeroom", "laundry room"]
BILLS = ["rent", "electricity bill", "water bill", "phone bill", "internet bill", "parking fine",
         "gym membership", "school fees", "insurance", "invoice from the supplier", "gas bill",
         "repair bill"]
FACILITIES = ["pool", "gym", "library", "canteen", "cafeteria", "museum", "post office",
              "car park", "playground", "sports hall", "computer room", "reading room",
              "bike shop", "pharmacy", "bakery", "laundry room", "study room", "tennis court"]
COUNT = ["two", "three", "four", "five", "six", "ten", "twelve", "twenty"]

# explanations
# optional second sentence that explains why (same in the polite version)
REASONS = {
    "cold": ["It's freezing in the {room}", "The wind is blowing my papers away", "It's raining outside",
             "The heating is on", "The baby is sleeping"],
    "hot": ["It's really hot in the {room}", "The room needs some fresh air", "It smells of paint in here",
            "Nobody can breathe in here"],
    "noise": ["I'm trying to work", "I have a call in ten minutes", "The baby is sleeping",
              "The neighbours complained", "I can't hear {name}", "It's after midnight",
              "I'm trying to study for my exam"],
    "energy": ["Nobody is using it", "The electricity bill is huge", "We are leaving soon",
               "It's been on all day"],
    "need": ["I need it for the {event}", "I can't find mine", "Mine is broken",
             "I'm leaving at {clock}", "We are already late", "I need it in ten minutes"],
    "shop": ["We are out of it", "There is none left", "The shop closes at {clock}",
             "We have guests on {day}", "We need it for the {event}"],
    "tidy": ["We have visitors tonight", "Your parents are coming on {day}", "I can't find anything",
             "The {room} is a mess", "The landlord is visiting on {day}"],
    "work": ["The client is waiting", "The deadline is {day}", "{name} needs it for the {event}",
             "Everyone is waiting", "The meeting starts at {clock}", "My manager asked for it"],
    "money": ["It was due on {day}", "We got a second reminder", "The landlord called again",
              "There is a late fee"],
    "stop": ["It disturbs the whole team", "I asked you last week", "I have mentioned it twice already",
             "Other people have noticed it too", "It is becoming a problem", "It makes it hard to focus"],
}

# (base, -ing, past participle, complement, reason group)
ACTIONS += [
    ("close", "closing", "closed", "the {openable}", "cold"),
    ("open", "opening", "opened", "the {openable}", "hot"),
    ("lock", "locking", "locked", "the {openable} before you leave", None),
    ("turn", "turning", "turned", "off the {device}", "energy"),
    ("turn", "turning", "turned", "on the {device}", None),
    ("turn", "turning", "turned", "off the lights in the {room}", "energy"),
    ("turn", "turning", "turned", "the {loud} down", "noise"),
    ("switch", "switching", "switched", "off the {device}", "energy"),
    ("bring", "bringing", "brought", "me the {obj}", "need"),
    ("bring", "bringing", "brought", "me {food}", None),
    ("get", "getting", "gotten", "me {food} on your way back", None),
    ("hand", "handing", "handed", "me the {obj}", "need"),
    ("pass", "passing", "passed", "me the {obj}", None),
    ("give", "giving", "given", "me back my {obj}", "need"),
    ("put", "putting", "put", "the {obj_p} away", "tidy"),
    ("put", "putting", "put", "the {obj} back in the {room}", "tidy"),
    ("pick", "picking", "picked", "up the {obj_p} from the {room}", "tidy"),
    ("carry", "carrying", "carried", "the {obj_p} to the {room}", None),
    ("take", "taking", "taken", "out the bins", "tidy"),
    ("take", "taking", "taken", "the {obj_p} upstairs", "tidy"),
    ("buy", "buying", "bought", "more {supplies}", "shop"),
    ("buy", "buying", "bought", "{supplies} on your way home", "shop"),
    ("order", "ordering", "ordered", "more {supplies}", "shop"),
    ("pay", "paying", "paid", "the {bill}", "money"),
    ("clean", "cleaning", "cleaned", "the {room}", "tidy"),
    ("tidy", "tidying", "tidied", "up the {room}", "tidy"),
    ("vacuum", "vacuuming", "vacuumed", "the {room}", "tidy"),
    ("empty", "emptying", "emptied", "the dishwasher", "tidy"),
    ("do", "doing", "done", "the dishes", "tidy"),
    ("wash", "washing", "washed", "the {obj_p}", "tidy"),
    ("wash", "washing", "washed", "the car", None),
    ("feed", "feeding", "fed", "the cat", None),
    ("walk", "walking", "walked", "the dog", None),
    ("set", "setting", "set", "the table for dinner", None),
    ("fix", "fixing", "fixed", "the {device}", None),
    ("find", "finding", "found", "my {obj}", "need"),
    ("charge", "charging", "charged", "the {obj_s}", None),
    ("make", "making", "made", "{count} copies of the {doc}", "work"),
    ("write", "writing", "written", "the {doc} for the {event}", "work"),
    ("finish", "finishing", "finished", "the {doc} for {name}", "work"),
    ("send", "sending", "sent", "the {doc} to {name}", "work"),
    ("fill", "filling", "filled", "in the {doc}", "work"),
    ("pick", "picking", "picked", "{name} up from the station", None),
    ("drop", "dropping", "dropped", "the {obj_p} at {name}'s place", None),
    ("lend", "lending", "lent", "me your {obj_s} for the {event}", "need"),
    ("save", "saving", "saved", "me a seat at the {event}", None),
]
# v1 actions about work documents use the "work" reasons
WORK_VERBS = {"send", "check", "review", "update", "finish", "prepare", "share", "submit", "sign",
              "print", "upload", "translate", "explain"}

ACTIVITIES += [
    ("lend you my {obj_s}", "lending you my {obj_s}"),
    ("lend you my {obj_p}", "lending you my {obj_p}"),
    ("cook dinner for everyone", "cooking dinner for everyone"),
    ("clean the {room} for you", "cleaning the {room} for you"),
    ("pay for the {event}", "paying for the {event}"),
    ("walk your dog", "walking your dog"),
    ("feed your cat while you are away", "feeding your cat while you are away"),
    ("carry your {obj_p} upstairs", "carrying your {obj_p} upstairs"),
    ("wait for you after the {event}", "waiting for you after the {event}"),
    ("go to the {event} with you", "going to the {event} with you"),
    ("buy the tickets for the {event}", "buying the tickets for the {event}"),
    ("fix your {device}", "fixing your {device}"),
    ("share my {food_n}", "sharing my {food_n}"),
    ("swap seats with you", "swapping seats with you"),
    ("host the {event} at my place", "hosting the {event} at my place"),
    ("check your homework", "checking your homework"),
    ("write the {doc} for you", "writing the {doc} for you"),
    ("drive {name} to the {facility}", "driving {name} to the {facility}"),
]

BEHAVIOURS += [
    "shouting across the {room}", "leaving the lights on", "leaving the {openable} open",
    "slamming the {openable}", "using my {obj_s} without asking", "eating my {food_n}",
    "kicking the back of my seat", "texting during the {event}", "smoking near the entrance",
    "leaving dirty dishes in the {room}", "parking in front of the gate",
    "playing games during work hours", "putting your feet on the table",
    "talking on the phone in the {place}", "throwing your clothes on the floor",
    "making noise after midnight", "leaving the TV on all night", "taking my {obj_p}",
    "borrowing my {obj_p} without asking", "ignoring my messages", "changing the channel",
    "playing the drums at night", "leaving your {obj_p} on the stairs",
    "taking the last coffee without making more", "using all the hot water",
    "turning the {loud} up", "running in the corridor", "eating in the {place}",
]

THINGS.update({
    "food": ["soup", "cake", "pasta", "sauce", "coffee", "salad", "curry", "pizza", "bread", "stew"],
    "art": ["painting", "drawing", "photo", "sketch", "mural", "collage", "sculpture"],
    "music": ["song", "melody", "solo", "performance", "recording"],
})
THINGS["text"] += ["translation", "homework", "answer", "notes", "description", "instructions",
                   "story", "poem", "review", "letter"]
THINGS["visual"] += ["layout", "website design", "photo album", "label", "banner", "invitation"]
THINGS["plan"] += ["seating plan", "travel plan", "menu plan", "timetable"]
ISSUES += [
    ("way too salty", "is a little salty", {"food"}),
    ("way too sweet", "is a little sweet", {"food"}),
    ("way too spicy", "is a bit spicy", {"food"}),
    ("cold", "is a bit cold", {"food"}),
    ("burnt", "is a little overcooked", {"food"}),
    ("tasteless", "could use a bit more flavour", {"food"}),
    ("messy", "looks a little messy", {"art", "visual"}),
    ("way too dark", "is a little dark", {"art"}),
    ("really boring", "is not very lively yet", {"art", "music"}),
    ("out of tune", "is a little out of tune", {"music"}),
    ("way too loud", "is a little loud", {"music"}),
    ("way too fast", "is a little fast", {"music"}),
    ("completely off topic", "does not quite stay on topic", {"text"}),
    ("all blurry", "is a bit blurry", {"art", "visual"}),
    ("badly translated", "could be translated a bit better", {"text"}),
]

P.update({
    "no_hot_water": ("has no hot water", "present"), "cold": ("is cold", "present"),
    "dirty": ("is dirty", "present"), "not_cleaned": ("has not been cleaned", "perfect"),
    "smells": ("smells of smoke", "present"), "lost": ("was lost", "past"),
    "missing": ("is missing", "present"), "overbooked": ("was overbooked", "past"),
    "out_of_order": ("is out of order", "present"), "leaking": ("keeps leaking", "present"),
    "arrived_cold": ("arrived cold", "past"), "given_away": ("was given to someone else", "past"),
    "too_expensive": ("cost twice the usual price", "past"), "stuck": ("has been stuck", "perfect"),
})
SERVICES.update({
    "hotel room": [P[k] for k in ("no_hot_water", "dirty", "not_cleaned", "smells", "noisy")],
    "shower": [P[k] for k in ("no_hot_water", "not_working", "broken", "leaking")],
    "apartment": [P[k] for k in ("no_hot_water", "cold", "not_cleaned")],
    "meal": [P[k] for k in ("cold", "arrived_cold", "late", "missing")],
    "pizza": [P[k] for k in ("arrived_cold", "late", "not_arrived")],
    "coffee": [P[k] for k in ("cold",)],
    "bus": [P[k] for k in ("late", "cancelled", "delayed")],
    "taxi": [P[k] for k in ("late", "not_arrived")],
    "luggage": [P[k] for k in ("lost", "not_arrived", "damaged")],
    "seat": [P[k] for k in ("given_away", "broken", "dirty")],
    "appointment": [P[k] for k in ("cancelled", "changed")],
    "table reservation": [P[k] for k in ("cancelled", "disappeared", "overbooked")],
    "wifi": [P[k] for k in ("not_working", "cutting_out", "slow")],
    "fridge": [P[k] for k in ("not_working", "noisy", "broken", "leaking")],
    "car repair": [P[k] for k in ("late", "not_finished", "too_expensive")],
    "ticket": [P[k] for k in ("charged_twice", "not_arrived", "missing")],
    "dishwasher": [P[k] for k in ("broken", "leaking", "not_working")],
    "vending machine": [P[k] for k in ("out_of_order", "broken", "stuck")],
    "parcel": [P[k] for k in ("not_arrived", "wrong_address", "damaged", "lost")],
    "toilet": [P[k] for k in ("out_of_order", "leaking", "dirty")],
    "gym locker": [P[k] for k in ("stuck", "broken")],
    "online order": [P[k] for k in ("late", "missing", "charged_twice", "wrong_address")],
})
DURATIONS["perfect"] += ["for the whole week", "since Monday", "for three hours"]
NUMBERED_SERVICES |= {"table reservation", "parcel", "online order", "ticket"}

FACTS += [
    "the {facility} opens at {clock}", "{name} already paid the {bill}", "the {bill} is due on {day}",
    "the {obj_p} are in the {room}", "the {obj_s} is in the {room}", "my flight lands at {clock}",
    "the {event} is in the {place}", "the parcel arrived on {day}", "{name} wrote the {doc}",
    "the {room} was cleaned this morning", "the {device} was fixed on {day}",
    "the {facility} is closed on {day}", "we need {count} chairs for the {event}",
    "the {event} costs {price}", "{name} is picking up the cake", "the train to {city} is on platform {count}",
    "the {doc} has {count} pages", "the dog was walked this morning", "it's {name}'s turn to cook",
    "the {obj_s} belongs to {name}", "the tickets for the {event} are sold out",
]

QUESTIONS += [
    ("why is the {openable} open", "why the {openable} is open"),
    ("who left the {openable} open", "who left the {openable} open"),
    ("what's wrong with the {device}", "what is wrong with the {device}"),
    ("who moved my {obj_s}", "who moved my {obj_s}"),
    ("where are my {obj_p}", "where my {obj_p} are"),
    ("where is my {obj_s}", "where my {obj_s} is"),
    ("when is the {bill} due", "when the {bill} is due"),
    ("how much is the {bill} this month", "how much the {bill} is this month"),
    ("who ate my {food_n}", "who ate my {food_n}"),
    ("why is the {room} so cold", "why the {room} is so cold"),
    ("where did you park the car", "where you parked the car"),
    ("what time do we leave for {city}", "what time we leave for {city}"),
    ("who is cooking dinner tonight", "who is cooking dinner tonight"),
    ("when does the {facility} open on {day}", "when the {facility} opens on {day}"),
    ("why didn't you call {name} back", "why you didn't call {name} back"),
    ("how many people are coming to the {event}", "how many people are coming to the {event}"),
    ("who used the last of the {supplies}", "who used the last of the {supplies}"),
    ("where is the {facility}", "where the {facility} is"),
    ("which bus goes to the {facility}", "which bus goes to the {facility}"),
    ("when are you going to fix the {device}", "when you are going to fix the {device}"),
    ("what are we having for dinner", "what we are having for dinner"),
    ("why is the {device} still on", "why the {device} is still on"),
]

# facilities (pool, gym...) are announced like events
FACILITY_CHANGES = [
    "is closed today", "is closed for repairs until {day}", "will close at {clock} today",
    "opens one hour later on {day}", "is closed this weekend", "will be closed on {day}",
    "has new opening hours", "is only open to members this week", "is closed for cleaning on {day}",
    "will reopen on {day}", "is full for the rest of the day",
]
CHANGES += [
    "is sold out", "is cancelled because of the weather", "has been moved to the {facility}",
    "will be held outside", "now needs a ticket", "is limited to {count} people",
]

STOP_CONTEXT += ["at night", "when I'm cooking", "after 10 pm"]

# contexts for food / art / music (the v1 contexts only make sense for documents)
THING_CONTEXT_BY_KIND = {
    "food": ["", "for the {event}", "you made yesterday", "from the canteen", "at the {event}"],
    "art": ["", "for the {event}", "in the {room}", "for {name}", "you finished yesterday"],
    "music": ["", "for the {event}", "at the {event}", "you recorded yesterday"],
}
