import { useEffect, useState } from "react";
import Home from "./pages/Home";
import Evaluation from "./pages/Evaluation";
import Milestones from "./pages/Milestones";

function getPage() {
  const path = window.location.pathname.replace(/\/$/, "") || "/";

  if (path === "/evaluation") {
    return "evaluation";
  }

  if (path === "/milestones") {
    return "milestones";
  }

  return "home";
}

function App() {
  const [page, setPage] = useState(getPage);

  useEffect(() => {
    const handleNavigation = () => {
      setPage(getPage());
      window.scrollTo(0, 0);
    };

    window.addEventListener("popstate", handleNavigation);

    return () => {
      window.removeEventListener("popstate", handleNavigation);
    };
  }, []);

  useEffect(() => {
    if (page === "evaluation") {
      document.title = "DermaLens | Evaluation";
    } else if (page === "milestones") {
      document.title = "DermaLens | Milestones";
    } else {
      document.title = "DermaLens";
    }
  }, [page]);

  if (page === "evaluation") {
    return <Evaluation />;
  }

  if (page === "milestones") {
    return <Milestones />;
  }

  return <Home />;
}

export default App;