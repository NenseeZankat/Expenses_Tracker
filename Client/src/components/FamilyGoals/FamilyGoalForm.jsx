


import React, { useState, useEffect } from "react";
import { useSelector } from "react-redux";
import axios from "axios";

const CreateFamilyGoal = () => {
  const [familyId, setFamilyId] = useState(null);
  const [formData, setFormData] = useState({
    goalName: "",
    targetAmount: "",
    startDate: "",
    endDate: "",
    reward: "",
  });

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Get user and token from Redux or LocalStorage
  const user = useSelector((state) => state.auth.user);
  const storedUser = JSON.parse(localStorage.getItem("userInfo"));
  const userId = user?.id || storedUser?.id;
  const token = user?.token || storedUser?.token;

  // Fetch full user profile & extract familyId
  useEffect(() => {
    const fetchUserProfile = async () => {
      if (!userId || !token) {
        console.error("❌ No userId or token found!");
        return;
      }

      try {
        const response = await axios.get("https://backend-johh.onrender.com/api/v1/users/profile", {
          headers: { Authorization: `Bearer ${token}` }, // Attach token here
        });

        const userProfile = response.data;
        if (userProfile?.familyId) {
          setFamilyId(userProfile.familyId);
          console.log("✅ Family ID:", userProfile.familyId);
        } else {
          console.warn("⚠️ No familyId found for this user.");
        }
      } catch (error) {
        console.error("❌ Error fetching user profile:", error);
      }
    };

    fetchUserProfile();
  }, [userId, token]);

  // Handle input changes
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage("");

    if (!familyId) {
      setMessage("❌ Family ID is missing. Try again later.");
      setLoading(false);
      return;
    }

    try {
      await axios.post(
        "https://backend-johh.onrender.com/api/family-goals/",
        { ...formData, familyId },
        { headers: { Authorization: `Bearer ${token}` } } // Include token here
      );

      //setMessage("✅ Family goal created successfully!");
      setFormData({
        goalName: "",
        targetAmount: "",
        startDate: "",
        endDate: "",
        reward: "",
      });
    } catch (error) {
      console.error("❌ Error creating goal:", error);
      setMessage("❌ Failed to create goal. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={containerStyle}>
      <h2 style={{ textAlign: "center", color: "#007bff" }}>🎯 Create a Family Goal</h2>

      {message && <p style={{ textAlign: "center", color: message.includes("✅") ? "green" : "red" }}>{message}</p>}
      {familyId ? <p>✅ Family ID: {familyId}</p> : <p>⏳ Loading Family ID...</p>}

      <form onSubmit={handleSubmit} style={formStyle}>
        <label>Goal Name:</label>
        <input type="text" name="goalName" value={formData.goalName} onChange={handleChange} required style={inputStyle} />

        <label>Target Amount (₹):</label>
        <input type="number" name="targetAmount" value={formData.targetAmount} onChange={handleChange} required style={inputStyle} />

        <label>Start Date:</label>
        <input type="date" name="startDate" value={formData.startDate} onChange={handleChange} required style={inputStyle} />

        <label>End Date:</label>
        <input type="date" name="endDate" value={formData.endDate} onChange={handleChange} required style={inputStyle} />

        <label>Reward (optional):</label>
        <input type="text" name="reward" value={formData.reward} onChange={handleChange} style={inputStyle} />

        <button type="submit" disabled={loading || !familyId} style={buttonStyle}>
          {loading ? "Creating..." : "Create Goal"}
        </button>
      </form>
    </div>
  );
};

// Styles
const containerStyle = {
  maxWidth: "500px",
  margin: "auto",
  padding: "20px",
  background: "#f8f9fa",
  borderRadius: "8px",
  boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)",
};

const formStyle = {
  display: "flex",
  flexDirection: "column",
};

const inputStyle = {
  padding: "10px",
  marginBottom: "10px",
  borderRadius: "5px",
  border: "1px solid #ccc",
  fontSize: "16px",
};

const buttonStyle = {
  padding: "10px",
  background: "#007bff",
  color: "#fff",
  border: "none",
  borderRadius: "5px",
  fontSize: "16px",
  cursor: "pointer",
};

export default CreateFamilyGoal;
