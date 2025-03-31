class UserQuery:
    def __init__(self, user='test', query='', max_tokens=200):
        if query == '' or query is None:
            raise ValueError('Query cannot be empty')
        
        self.user = user
        self.query = self.process(query)
        self.max_tokens = max_tokens

    def process(self, query):
        clean_query = self.handle_attack(query.lower().strip()[:self.max_tokens])
        return clean_query