;
import { 
  Activity, 
  ShieldCheck, 
  Eye, 
  Sparkles, 
  Palette, 
  Gauge, 
  Layers, 
  TrendingUp 
} from 'lucide-react';

const metricCategories = [
  {
    category: "Image Quality & Structural Metrics",
    description: "Evaluates standard mathematical signal quality and spatial fidelity against raw input.",
    items: [
      {
        name: "Laplacian Variance",
        symbol: "Var_Lap",
        icon: Eye,
        badge: "Blur Detection",
        purpose: "Measures high-frequency gradient density across the image to evaluate relative sharpness and detect motion/focus blur.",
        essential: "Dermatological feature extraction (e.g., fine pigment networks and border irregularity) relies on high edge contrast. Unsharp candidates are rejected to prevent diagnostic blurring."
      },
      {
        name: "PSNR (Peak Signal-to-Noise Ratio)",
        symbol: "PSNR",
        icon: Activity,
        badge: "Signal Fidelity",
        purpose: "Quantifies the ratio between maximum potential signal strength and corrupting reconstruction noise.",
        essential: "Ensures model enhancement steps (such as super-resolution or deblurring) do not introduce excessive pixel distortion or compression artifacts."
      },
      {
        name: "SSIM (Structural Similarity Index)",
        symbol: "SSIM",
        icon: Layers,
        badge: "Structural Integrity",
        purpose: "Evaluates degradation by analyzing structural patterns, luminance, and local contrast similarity.",
        essential: "Prevents deep learning generators (like MPRNet or EDSR) from hallucinating non-existent skin features or distorting natural skin texture."
      },
      {
        name: "RMSE (Root Mean Square Error)",
        symbol: "RMSE",
        icon: Gauge,
        badge: "Discrepancy Measure",
        purpose: "Computes pixel-by-pixel intensity variance between original and processed output frames.",
        essential: "Acts as a hard mathematical baseline constraint to catch severe pixel-level variance before clinical analysis."
      }
    ]
  },
  {
    category: "Clinical & Domain-Specific Metrics",
    description: "Restricts quality analysis specifically to the isolated lesion region using localized segmentation masks.",
    items: [
      {
        name: "Lesion-Specific Sharpness / SSIM",
        symbol: "M_lesion ⊙ Quality",
        icon: Sparkles,
        badge: "Mask-Bounded Quality",
        purpose: "Calculates sharpness and structural similarity strictly inside the boundaries predicted by SegFormer.",
        essential: "Background healthy skin can distort global quality scores. Isolating the lesion ensures enhancement algorithms optimize the clinically relevant lesion tissue."
      },
      {
        name: "Color Shift / Color Difference",
        symbol: "ΔE",
        icon: Palette,
        badge: "Diagnostic Palette Safety",
        purpose: "Measures mean RGB/LAB color distance drift across raw and processed skin tissue.",
        essential: "Lesion color variations (e.g., erythema, pigmentation, vascularity) are primary diagnostic indicators. Uncontrolled color shifts can cause misclassification by downstream models."
      },
      {
        name: "Noise Estimation Variance",
        symbol: "σ_noise",
        icon: Activity,
        badge: "Artifact Rejection",
        purpose: "Analyzes standard deviation of high-frequency noise bands to prevent over-sharpening artifacts.",
        essential: "Algorithms like CLAHE can amplify background image noise. This metric flags over-processed outputs before they reach diagnostic classifiers."
      }
    ]
  },
  {
    category: "Downstream AI Reliability Metrics",
    description: "Evaluates real-time classification stability and spatial coverage.",
    items: [
      {
        name: "Classifier Confidence Delta",
        symbol: "ΔC",
        icon: TrendingUp,
        badge: "Model Certainty Shift",
        purpose: "Tracks change in softmax probability score output by the downstream classifier (e.g., ResNet-18) pre- and post-enhancement.",
        essential: "Verifies whether image enhancement genuinely clarifies diagnostic features and increases classifier confidence without causing class flips."
      },
      {
        name: "Lesion Area Coverage Ratio",
        symbol: "Area_mask / Area_total",
        icon: ShieldCheck,
        badge: "Spatial Normalization",
        purpose: "Computes ratio of segmented lesion area to the full resolution frame size.",
        essential: "Determines adaptive spatial bounding boxes and triggers fallbacks if the lesion is cropped, off-center, or excessively small."
      }
    ]
  }
];

export default function MetricsPage({ onClose }) {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-700 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Activity className="w-8 h-8 text-indigo-400" />
              DermaLens Reliability Metrics
            </h1>
            <p className="text-slate-400 mt-2 text-sm max-w-3xl">
              DermaLens evaluates each image candidate through multi-tier validation checks before passing enhanced images to downstream diagnostic classifiers. Below is a breakdown of all active metrics and why each is essential.
            </p>
          </div>
          {onClose && (
            <button 
              onClick={onClose}
              className="px-4 py-2 text-sm bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700"
            >
              Close View
            </button>
          )}
        </div>
      </div>

      {/* Metrics Categories */}
      <div className="space-y-10">
        {metricCategories.map((group, idx) => (
          <div key={idx} className="space-y-4">
            <div>
              <h2 className="text-xl font-semibold text-indigo-300">{group.category}</h2>
              <p className="text-xs text-slate-400">{group.description}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {group.items.map((item, itemIdx) => {
                const IconComponent = item.icon;
                return (
                  <div 
                    key={itemIdx}
                    className="bg-slate-900 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-all flex flex-col justify-between space-y-4"
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-3">
                          <div className="p-2.5 bg-indigo-950/60 rounded-lg text-indigo-400 border border-indigo-900/50">
                            <IconComponent className="w-5 h-5" />
                          </div>
                          <div>
                            <h3 className="font-semibold text-slate-100 text-base">{item.name}</h3>
                            <span className="text-xs font-mono text-indigo-400 bg-indigo-950/40 px-2 py-0.5 rounded border border-indigo-900/40">
                              {item.symbol}
                            </span>
                          </div>
                        </div>
                        <span className="text-xs bg-slate-800 text-slate-300 px-2.5 py-1 rounded-full border border-slate-700 whitespace-nowrap">
                          {item.badge}
                        </span>
                      </div>

                      <div className="mt-4 space-y-2">
                        <div>
                          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Purpose</p>
                          <p className="text-sm text-slate-300 mt-0.5">{item.purpose}</p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
                      <p className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5 mb-1">
                        <ShieldCheck className="w-3.5 h-3.5" />
                        Why it is Essential
                      </p>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {item.essential}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}