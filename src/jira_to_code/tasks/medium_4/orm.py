from django.db.models import Prefetch
class User:
    def __init__(self, id): self.id = id
class DB:
    def get_users(self): return [User(i) for i in range(50)]
    def get_profile(self, user): return {'profile': 'data'}
    def get_profile_list_prefetch(self, users):
        return [
            {
                'user_id': user.id,
                'profile': self.get_profile(user),
            }
            for user in users
        ]
def fetch_user_profiles(db):
    return db.get_profile_list_prefetch(db.get_users())