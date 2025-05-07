

import React, { useState } from "react";
import axios from "axios";

const TaxReport = ({ userId }) => {
  const [loading, setLoading] = useState(false);

const handleDownloadReport = async () => {
    if (!userId) {
        console.error("Error: User ID is missing!");
        return;
    }

    try {
        setLoading(true);
        console.log("Generating and opening report for user:", userId);
        const currentYear = new Date().getFullYear();

        // ✅ Open new tab early to avoid browser blocking
        const newTab = window.open("", "_blank");

        const response = await axios.get(`https://backend-johh.onrender.com/api/reports/generate/${userId}/${currentYear}`, {
            responseType: "blob", // Ensure correct file handling
            headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
        });

        if (response.status === 200) {
            const blob = new Blob([response.data], { type: "application/pdf" });
            const url = window.URL.createObjectURL(blob);

            // ✅ Redirect the new tab to the generated PDF
            newTab.location.href = url;
        } else {
            console.error("Failed to generate report");
            newTab.close(); // Close the tab if error occurs
        }
    } catch (error) {
        console.error("Error fetching tax report:", error.response ? error.response.data : error.message);
    } finally {
        setLoading(false);
    }
};

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold">Download Tax Report</h2>
      <button
        onClick={handleDownloadReport}
        className={`mt-3 px-4 py-2 rounded ${loading ? "bg-gray-400" : "bg-blue-500 text-white"}`}
        disabled={loading}
      >
        {loading ? "Generating..." : "Download Report"}
      </button>
    </div>
  );
};

export default TaxReport;
 