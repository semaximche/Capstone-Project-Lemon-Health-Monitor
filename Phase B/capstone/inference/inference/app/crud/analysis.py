from sqlalchemy.orm import Session
from inference.app.db.models import Analysis


class AnalysisCRUD:
    def __init__(self):
        pass

    def create(self, db: Session, data) -> str:
        obj = Analysis(
            user_id=data["user_id"],
            presigned_url=data["presigned_url"],
            description=data["description"],
            summary=data["summary"],

        )
        db.add(obj)
        db.commit()
        db.refresh(obj)

        return obj.id

    def get(self, db: Session, analysis_id: str) -> Analysis | None:
        return db.query(Analysis).filter(Analysis.id == analysis_id).first()

    def delete(self, db: Session, analysis_id: str) -> bool:
        obj = self.get(db, analysis_id)
        if not obj:
            return False
        db.delete(obj)
        db.commit()
        return True



analysis_crud = AnalysisCRUD()