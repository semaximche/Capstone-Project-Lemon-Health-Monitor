// Server analysis request schema
interface AnalysisRequest {
  image: File;
}

// Server analysis response schema code 201
interface AnalysisResponseSuccess {
  analysisID: String;
  status: String;
  description: String;
}

// Server get analysis request schema
interface GetAnalysisRequest {
  analysisID: String;
}

// Server get analysis response schema code 200
interface GetAnalysisResponseSucess {
  analysisID: String;
  status: String;
  description: String;
}