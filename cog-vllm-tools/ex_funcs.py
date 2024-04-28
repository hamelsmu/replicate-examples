tools_good_json = '[\
{"name": "get_exchange_rate", "description": "Get the exchange rate between two currencies", "parameters": {"type": "object", "properties": {"base_currency": {"type": "string", "description": "The currency to convert from"}, "target_currency": {"type": "string", "description": "The currency to convert to"}}, "required": ["base_currency", "target_currency"]}},\
{"name": "create_contact", "description": "Create a new contact", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "The name of the contact"}, "email": {"type": "string", "description": "The email address of the contact"}}, "required": ["name", "email"]}}\
]'


tools_bad_json = '[\
{name: "get_exchange_rate", "description": "Get the exchange rate between two currencies", "parameters": {"type": "object", "properties": {"base_currency": {"type": "string", "description": "The currency to convert from"}, "target_currency": {"type": "string", "description": "The currency to convert to"}}, "required": ["base_currency", "target_currency"]}},\
]'