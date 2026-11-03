import React from "react";
import ReactDOM from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./styles.css";
import App from "./App";
import { useStudio } from "./store";

// The store is reachable from the browser console for inspection and scripting.
(window as unknown as { studio: typeof useStudio }).studio = useStudio;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
