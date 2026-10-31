"""Shared constants: category taxonomy, merchant knowledge base and display settings."""

CATEGORIES = [
    "Rent", "Utilities", "Groceries", "Food Delivery", "Dining", "Transport",
    "Fuel", "Shopping", "Entertainment", "Subscriptions", "Health", "Travel",
    "Education", "Miscellaneous",
]

NEEDS = {"Rent", "Utilities", "Groceries", "Transport", "Fuel", "Health", "Education"}
WANTS = set(CATEGORIES) - NEEDS

MERCHANTS = {
    "Rent": ["House Rent"],
    "Utilities": ["BESCOM Electricity", "ACT Fibernet", "Jio Recharge"],
    "Groceries": ["BigBasket", "Blinkit", "DMart", "Zepto"],
    "Food Delivery": ["Swiggy", "Zomato"],
    "Dining": ["Barbeque Nation", "Starbucks", "Cafe Coffee Day", "Domino's"],
    "Transport": ["Uber", "Ola", "Namma Metro", "Rapido"],
    "Fuel": ["Indian Oil", "HP Petrol Pump", "Shell"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio"],
    "Entertainment": ["BookMyShow", "PVR Cinemas", "Steam"],
    "Subscriptions": ["Netflix", "Spotify", "Disney Hotstar", "YouTube Premium", "Cult Fit Gym"],
    "Health": ["Apollo Pharmacy", "PharmEasy", "1mg", "Practo"],
    "Travel": ["MakeMyTrip", "IRCTC", "IndiGo", "RedBus"],
    "Education": ["Udemy", "Coursera", "Crossword Books"],
    "Miscellaneous": ["Local Store", "ATM Cash Withdrawal", "Gift Shop"],
}

MERCHANT_CATEGORY = {m: c for c, ms in MERCHANTS.items() for m in ms}

KEYWORD_RULES = [
    ("PHARMA", "Health"), ("HOSPITAL", "Health"), ("CLINIC", "Health"), ("MEDIC", "Health"),
    ("PETROL", "Fuel"), ("FUEL", "Fuel"), ("DIESEL", "Fuel"),
    ("RESTAURANT", "Dining"), ("CAFE", "Dining"), ("COFFEE", "Dining"), ("PIZZA", "Dining"),
    ("SUPERMARKET", "Groceries"), ("GROCER", "Groceries"), ("MART", "Groceries"),
    ("CINEMA", "Entertainment"), ("MOVIE", "Entertainment"),
    ("AIRLINES", "Travel"), ("RAILWAY", "Travel"), ("HOTEL", "Travel"),
    ("ELECTRIC", "Utilities"), ("BROADBAND", "Utilities"), ("RECHARGE", "Utilities"),
    ("COURSE", "Education"), ("SCHOOL", "Education"), ("COLLEGE", "Education"),
    ("SUBSCRIPTION", "Subscriptions"), ("RENT", "Rent"),
    ("CAB", "Transport"), ("METRO", "Transport"), ("TAXI", "Transport"),
]

CATEGORY_NOTES = {
    "Rent": ["house rent"],
    "Utilities": ["electricity bill", "broadband bill", "mobile recharge"],
    "Groceries": ["groceries", "vegetables", "monthly grocery", "fruits and milk"],
    "Food Delivery": ["food order", "dinner", "lunch", "biryani", "late night snack"],
    "Dining": ["dinner with friends", "coffee", "brunch", "team lunch"],
    "Transport": ["cab ride", "trip to office", "airport drop", "metro recharge"],
    "Fuel": ["petrol", "fuel refill", "diesel"],
    "Shopping": ["clothes", "order shoes", "gadget accessory", "home decor"],
    "Entertainment": ["movie tickets", "game purchase", "concert"],
    "Subscriptions": ["subscription renewal", "monthly plan"],
    "Health": ["medicines", "pharmacy bill", "doctor consultation", "lab test"],
    "Travel": ["flight ticket", "train booking", "hotel stay", "bus ticket"],
    "Education": ["online course", "book purchase", "exam fee"],
    "Miscellaneous": ["gift", "cash", "local store", "repair"],
}

PAYMENT_MODES = ["UPI", "Card", "NetBanking", "Cash"]

PALETTE = [
    "#2F6FDE", "#E8743B", "#19A979", "#945ECF", "#E6B800", "#D1495B", "#13A8A8",
    "#8C564B", "#5B8FF9", "#7A8B99", "#F28BB5", "#4E9F3D", "#C77DFF", "#B08968",
]
CATEGORY_COLORS = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(CATEGORIES)}
CURRENCY = "₹"
