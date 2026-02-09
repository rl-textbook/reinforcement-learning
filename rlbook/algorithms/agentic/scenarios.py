"""Customer support email scenario dataset (Chapter 7)."""

EMAIL_SCENARIOS = [
    {
        'id': 0, 'customer_id': 'C001',
        'email': "I'm furious! My order #1234 never arrived and I need it urgently!",
        'truth': {'status': 'shipped_delayed', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 4, 6], 'is_repeat': False,
    },
    {
        'id': 1, 'customer_id': 'C002',
        'email': "The product arrived damaged. Can I get a replacement?",
        'truth': {'status': 'delivered', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [4, 7], 'is_repeat': False,
    },
    {
        'id': 2, 'customer_id': 'C003',
        'email': "I want to cancel my subscription immediately.",
        'truth': {'status': 'active', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [5, 8], 'is_repeat': False,
    },
    {
        'id': 3, 'customer_id': 'C004',
        'email': "Charged twice for the same order! Fix this now!",
        'truth': {'status': 'duplicate_charge', 'payment': 'double_charged', 'fraud_risk': 'low'},
        'optimal_actions': [3, 6], 'is_repeat': False,
    },
    {
        'id': 4, 'customer_id': 'C005',
        'email': "My account was hacked and someone made unauthorized purchases.",
        'truth': {'status': 'fraudulent', 'payment': 'completed', 'fraud_risk': 'high'},
        'optimal_actions': [2, 3, 10], 'is_repeat': False,
    },
    {
        'id': 5, 'customer_id': 'C006',
        'email': "Great service! Just wanted to say thanks.",
        'truth': {'status': 'satisfied', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [0], 'is_repeat': False,
    },
    {
        'id': 6, 'customer_id': 'C007',
        'email': "The product description was misleading. I want my money back.",
        'truth': {'status': 'mismatch', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 1, 6], 'is_repeat': False,
    },
    {
        'id': 7, 'customer_id': 'C008',
        'email': "I received someone else's order by mistake.",
        'truth': {'status': 'wrong_item', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 4, 7], 'is_repeat': False,
    },
    {
        'id': 8, 'customer_id': 'C009',
        'email': "Where is my refund? It's been 3 weeks!",
        'truth': {'status': 'refund_pending', 'payment': 'refund_initiated', 'fraud_risk': 'low'},
        'optimal_actions': [3, 6], 'is_repeat': False,
    },
    {
        'id': 9, 'customer_id': 'C010',
        'email': "The item is defective and stopped working after 2 days.",
        'truth': {'status': 'defective', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [1, 7], 'is_repeat': False,
    },
    # Repeat customer scenarios
    {
        'id': 10, 'customer_id': 'C001',
        'email': "STILL waiting for my order! This is unacceptable!",
        'truth': {'status': 'shipped_delayed', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [9, 3, 4, 6], 'is_repeat': True,
    },
    {
        'id': 11, 'customer_id': 'C002',
        'email': "The replacement you sent is ALSO damaged! What's going on?",
        'truth': {'status': 'delivered', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [9, 4, 7], 'is_repeat': True,
    },
    {
        'id': 12, 'customer_id': 'C004',
        'email': "I'm still seeing the duplicate charge on my statement!",
        'truth': {'status': 'duplicate_charge', 'payment': 'double_charged', 'fraud_risk': 'low'},
        'optimal_actions': [9, 3, 6], 'is_repeat': True,
    },
    {
        'id': 13, 'customer_id': 'C011',
        'email': "I've been a loyal customer for 5 years, but this experience is terrible.",
        'truth': {'status': 'vip_complaint', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 0], 'is_repeat': False,
    },
    {
        'id': 14, 'customer_id': 'C012',
        'email': "The coupon code doesn't work. Can you apply it manually?",
        'truth': {'status': 'promo_issue', 'payment': 'pending', 'fraud_risk': 'low'},
        'optimal_actions': [3, 0], 'is_repeat': False,
    },
    {
        'id': 15, 'customer_id': 'C013',
        'email': "I returned the item but haven't received credit yet.",
        'truth': {'status': 'return_pending', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 4, 6], 'is_repeat': False,
    },
    {
        'id': 16, 'customer_id': 'C014',
        'email': "This is my 3rd time contacting you about the same issue!",
        'truth': {'status': 'repeat_issue', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [9, 3, 0], 'is_repeat': True,
    },
    {
        'id': 17, 'customer_id': 'C015',
        'email': "I need an invoice for my company's accounting.",
        'truth': {'status': 'invoice_request', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 0], 'is_repeat': False,
    },
    {
        'id': 18, 'customer_id': 'C016',
        'email': "The tracking shows delivered but I never received the package.",
        'truth': {'status': 'lost_package', 'payment': 'completed', 'fraud_risk': 'medium'},
        'optimal_actions': [3, 4, 2, 7], 'is_repeat': False,
    },
    {
        'id': 19, 'customer_id': 'C017',
        'email': "I accidentally ordered the wrong size. Can I exchange it?",
        'truth': {'status': 'exchange_needed', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 1, 7], 'is_repeat': False,
    },
]
