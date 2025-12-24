import { useEffect, useState } from "react";
import { useAuth } from "~/provider/auth-context";

export default function Dashboard() {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [atoken, setAtoken] = useState<string>('');

    const { token } = useAuth();

    useEffect((() => {
        setAtoken(token.value);
    }), [token, atoken])

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
        setFile(e.target.files[0]);
        setError(null);
        }
    };

    const handleUpload = async () => {
        if (!file) {
        setError("Please select an image first.");
        return;
        }

        const formData = new FormData();
        formData.append("image", file);

        try {
            setLoading(true);
            setError(null);
            console.log(atoken.value);
            const response = await fetch("http://127.0.0.1:8000/v1/analysis", {
                method: "POST",
                headers: {
                Accept: "application/json",
                Authorization: `Bearer ` + atoken,
                },
                body: formData,
            });

            if (!response.ok) {
                const text = await response.text();
                throw new Error(text || "Request failed");
            }

            const data = await response.json();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setLoading(false);
        }
    };


    return (
    <div className="min-h-screen bg-black p-6">
      <div className="mx-auto max-w-3xl space-y-6">
        <div className="flex aspect-square items-center justify-center rounded-xl bg-black shadow border-2 border-white text-white">
          <div className="w-full max-w-sm space-y-4 p-6 text-center">
            <h2 className="text-xl font-semibold">Upload Image</h2>
        
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm file:mr-4 file:rounded-lg file:border-0
                         file:bg-blue-600 file:px-4 file:py-2
                         file:text-white hover:file:bg-blue-700"/>

            <button
              onClick={handleUpload}
              disabled={loading}
              className="w-full rounded-lg bg-blue-600 px-4 py-2
                         font-medium text-white hover:bg-blue-700
                         disabled:cursor-not-allowed disabled:opacity-50">
              {loading ? "Analyzing..." : "Upload & Analyze"}
            </button>

            {error && <p className="text-sm text-red-600">{error}</p>}
          </div>
        </div>

        <div className="flex aspect-square items-center justify-center rounded-xl bg-black shadow border-2 border-white text-white ">
          <div className="w-full p-6">
            <h2 className="mb-4 text-xl font-semibold">Results</h2>

            {!loading && (<p className="text-gray-500">No results yet.</p>)}

            {loading && (<p className="text-gray-500">Processing image…</p>)}
          </div>
        </div>

      </div>
    </div>
  );
}