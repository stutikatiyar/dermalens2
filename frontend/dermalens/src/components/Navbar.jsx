import { useEffect, useState } from "react";

function navigate(path) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function Navbar() {
  const [currentPath, setCurrentPath] = useState(
    window.location.pathname.replace(/\/$/, "") || "/"
  );

  useEffect(() => {
    const handleNavigation = () => {
      setCurrentPath(
        window.location.pathname.replace(/\/$/, "") || "/"
      );
    };

    window.addEventListener("popstate", handleNavigation);
    return () => window.removeEventListener("popstate", handleNavigation);
  }, []);

  const handleNavigation = (e, path) => {
    e.preventDefault();

    if (path !== currentPath) {
      navigate(path);
    }
  };

  const handleHowItWorks = () => {
    if (currentPath !== "/") {
      navigate("/");
      window.setTimeout(() => {
        document.getElementById("how-it-works")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 500);
      return;
    }

    document.getElementById("how-it-works")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  const linkClass = (path) =>
    `transition ${
      currentPath === path
        ? "text-cyan-300 font-semibold"
        : "text-slate-400 hover:text-white"
    }`;

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-cyan-500/10 bg-[#020617]/75 backdrop-blur-2xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-6 py-5">
        <a
          href="/"
          onClick={(e) => handleNavigation(e, "/")}
          className="flex shrink-0 items-center gap-3"
        >
          <div className="h-3 w-3 animate-pulse rounded-full bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.9)]" />
          <h1 className="text-xl font-semibold tracking-tight text-white">
            DermaLens
          </h1>
          <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-2 py-1 font-mono text-xs text-cyan-300">
            AI
          </span>
        </a>

        <div className="flex items-center gap-4 text-sm font-medium md:gap-7">
          <div className="hidden items-center gap-2 text-cyan-300 lg:flex">
            <div className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />
            <span>AI Restoration Active</span>
          </div>

          <a
            href="/"
            onClick={(e) => handleNavigation(e, "/")}
            className={linkClass("/")}
          >
            Analyze
          </a>

          <button
            type="button"
            onClick={handleHowItWorks}
            className="whitespace-nowrap text-slate-400 transition hover:text-cyan-300"
          >
            How It Works
          </button>

          {/* Added Metrics Navigation Link */}
          <a
            href="/metrics"
            onClick={(e) => handleNavigation(e, "/metrics")}
            className={linkClass("/metrics")}
          >
            Metrics
          </a>

          <a
            href="/evaluation"
            onClick={(e) => handleNavigation(e, "/evaluation")}
            className={linkClass("/evaluation")}
          >
            Evaluation
          </a>

          <a
            href="/milestones"
            onClick={(e) => handleNavigation(e, "/milestones")}
            className={linkClass("/milestones")}
          >
            Milestones
          </a>

          <a
            href="https://github.com/stutikatiyar/dermalens"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden text-slate-400 transition hover:text-white sm:block"
          >
            GitHub
          </a>

          <a
            href="/"
            onClick={(e) => handleNavigation(e, "/")}
            className="whitespace-nowrap rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-white shadow-[0_0_30px_rgba(34,211,238,0.25)] transition-all duration-300 hover:scale-105 md:px-5"
          >
            Launch Demo
          </a>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;