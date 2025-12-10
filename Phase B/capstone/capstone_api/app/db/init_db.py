from .db import Base, engine
import app.db.models


def init_db():

    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":

    init_db()