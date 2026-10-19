import { useState } from "react";
import HeroSection from "./HeroSection";
import BackgroundGlow from "./BackgroundGlow";
import AIProcessing from "./AIProcessing";
import ResultSection from "./ResultSection";
import UploadSection from "./UploadSection";
import Navbar from "./Navbar";
import HowItWorks from "./HowItWorks";
import { uploadImage } from "../api/api";

export default function UploadBox() {
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState(null);
  const [enhanced, setEnhanced] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [mode, setMode] = useState("");
  const [insights, setInsights] = useState([]);
  const [enhancementAccepted, setEnhancementAccepted] = useState(true);
  const [lesionBbox, setLesionBbox] = useState(null);
  const [comparison, setComparison] = useState([]);

  async function handleChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setEnhanced(null);
    setMetrics(null);
    setConfidence(null);
    setMode("");
    setInsights([]);
    setEnhancementAccepted(true);
    setLesionBbox(null);
    setComparison([]);
    setImage(URL.createObjectURL(file));

    try {
      const data = await uploadImage(file);

      console.log("BACKEND RESPONSE:", data);
      console.log("Enhancement accepted:", data.enhancement_accepted);
      console.log("Mode:", data.mode);
      console.log("Enhanced image exists:", Boolean(data.enhanced_image));

      if (!data.enhanced_image) {
        throw new Error("Backend response does not contain an enhanced image.");
      }

      setEnhanced(`data:image/jpeg;base64,${data.enhanced_image}`);
      setLesionBbox(data.segmentation?.bbox ?? null);
      setComparison(Array.isArray(data.comparison) ? data.comparison : []);
      setMetrics(data.metrics);
      setConfidence(data.confidence);
      setMode(data.mode || "");
      setInsights(data.interpretation || []);
      setEnhancementAccepted(
        data.enhancement_accepted !== undefined
          ? data.enhancement_accepted
          : true
      );
    } catch (err) {
      console.error("Image processing failed:", err);
      alert("Failed to process image. Check the browser console and backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen bg-[#020617] overflow-hidden">
      <BackgroundGlow />
      <Navbar />
      <HeroSection />
      <HowItWorks />

      {!loading && !enhanced ? (
        <UploadSection handleChange={handleChange} />
      ) : loading ? (
        <div className="relative z-10 px-6 pb-24 -mt-10">
          <div className="max-w-4xl mx-auto">
            <AIProcessing />
          </div>
        </div>
      ) : null}

      {!loading && enhanced && (
        <ResultSection
          image={image}
          enhanced={enhanced}
          metrics={metrics}
          confidence={confidence}
          mode={mode}
          insights={insights}
          enhancementAccepted={enhancementAccepted}
          lesionBbox={lesionBbox}
          comparison={comparison}
        />
      )}
    </div>
  );
}
