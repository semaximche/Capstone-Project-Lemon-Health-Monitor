# from pydantic import Field
# from typing import Optional
#
# from pydantic import BaseModel
#
# class AnalysisCreate(BaseModel):
#     presigned_url: str = Field(
#         ...,
#         max_length=500,
#         description="The pre-signed URL (or direct URL) where the source image is stored."
#     )
#     description: Optional[str] = Field(
#         None,
#         max_length=1000,
#         description="A user-provided description or detailed context for this analysis."
#     )