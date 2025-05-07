

// import React, { useEffect, useState } from "react";
// import axios from "axios";

// const FamilyGoals = ({ familyId, userId }) => {
//   const [goals, setGoals] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [contributions, setContributions] = useState({}); // Track input values for each goal

//   useEffect(() => {
//     fetchGoals();
//   }, []);

//   const fetchGoals = async () => {
//     try {
//       const response = await axios.get(`http://localhost:8000/api/family-goals/${familyId}`);
//       console.log("API Response:", response.data);

//       const activeGoals = response.data.filter(goal => goal.currentAmount < goal.targetAmount);
//       setGoals(activeGoals);
//     } catch (error) {
//       console.error("Error fetching goals:", error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const joinChallenge = async (goalId) => {
//     try {
//       await axios.put(`http://localhost:8000/api/family-goals/${goalId}/join`, { userId });
//       fetchGoals();
//     } catch (error) {
//       console.error("Error joining challenge:", error);
//     }
//   };

//   const contributeAmount = async (goalId, amount) => {
//     const contributionAmount = Number(amount); // Convert input to number
  
//     if (isNaN(contributionAmount) || contributionAmount <= 0) {
//       alert("Please enter a valid amount!");
//       return;
//     }
  
//     try {
//       await axios.put(`http://localhost:8000/api/family-goals/${goalId}/contribute`, {
//         userId,
//         amount: contributionAmount,  // ✅ Send as a number
//       });
  
//       // Clear the input field after successful contribution
//       setContributions((prevContributions) => ({
//         ...prevContributions,
//         [goalId]: "",
//       }));
  
//       fetchGoals();
//     } catch (error) {
//       console.error("Error contributing:", error);
//     }
//   };
  
  

//   return (
//     <div style={{ maxWidth: "600px", margin: "20px auto", padding: "20px", borderRadius: "10px", background: "#f8f9fa", boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)" }}>
//       <h2 style={{ textAlign: "center", color: "#333" }}>Family Expense Challenges</h2>
//       {loading ? (
//         <p>Loading...</p>
//       ) : goals.length === 0 ? (
//         <p style={{ textAlign: "center", color: "green" }}>🎉 All goals are completed!</p>
//       ) : Array.isArray(goals) ? (
//         <ul style={{ listStyle: "none", padding: "0" }}>
//           {goals.map((goal) => (
//             <li key={goal._id} style={{ background: "#ffffff", padding: "15px", marginBottom: "10px", borderRadius: "8px", borderLeft: "5px solid #007bff", boxShadow: "2px 2px 6px rgba(0, 0, 0, 0.1)" }}>
//               <h3 style={{ fontSize: "18px", margin: "5px 0" }}>
//                 {goal.goalName} - ₹{goal.currentAmount} / ₹{goal.targetAmount}
//               </h3>
//               <p style={{ fontSize: "14px", color: "#555" }}>Status: {goal.status}</p>
//               <p style={{ fontSize: "14px", color: "#555" }}>Reward: {goal.reward}</p>
              
//               {goal.participants.some((p) => p.userId === userId) ? (
//                 <div style={{ marginTop: "10px" }}>
//                   <input
//                     type="number"
//                     placeholder="Enter amount"
//                     value={contributions[goal._id] || ""}
//                     onChange={(e) => setContributions({ ...contributions, [goal._id]: e.target.value })}
//                     style={{ padding: "8px", borderRadius: "5px", border: "1px solid #ccc", marginRight: "10px", width: "80px" }}
//                   />
//                   <button
//                     style={{ background: "#28a745", color: "white", border: "none", padding: "8px 12px", borderRadius: "5px", cursor: "pointer", transition: "0.3s", fontSize: "14px" }}
//                     onClick={() => contributeAmount(goal._id, contributions[goal._id])}
//                     onMouseOver={(e) => (e.target.style.background = "#218838")}
//                     onMouseOut={(e) => (e.target.style.background = "#28a745")}
//                   >
//                     Contribute
//                   </button>
//                 </div>
//               ) : (
//                 <button
//                   style={{ background: "#007bff", color: "white", border: "none", padding: "8px 12px", borderRadius: "5px", cursor: "pointer", transition: "0.3s", fontSize: "14px" }}
//                   onClick={() => joinChallenge(goal._id)}
//                   onMouseOver={(e) => (e.target.style.background = "#0056b3")}
//                   onMouseOut={(e) => (e.target.style.background = "#007bff")}
//                 >
//                   Join Challenge
//                 </button>
//               )}
//             </li>
//           ))}
//         </ul>
//       ) : (
//         <p>No goals available.</p>
//       )}
//     </div>
//   );
// };

// export default FamilyGoals;


import React, { useEffect, useState } from "react";
import axios from "axios";

const FamilyGoals = ({ familyId, userId }) => {
  const [goals, setGoals] = useState([]);
  const [pastGoals, setPastGoals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [contributions, setContributions] = useState({});
  const [showPastGoals, setShowPastGoals] = useState(false); // Toggle past goals

  useEffect(() => {
    fetchGoals();
  }, []);

  const fetchGoals = async () => {
    try {
      const response = await axios.get(`https://backend-johh.onrender.com/api/family-goals/${familyId}`);
      console.log("API Response:", response.data);

      const today = new Date();
      const activeGoals = response.data.filter(goal => goal.currentAmount < goal.targetAmount && new Date(goal.endDate) >= today);
      const expiredGoals = response.data.filter(goal => new Date(goal.endDate) < today || goal.currentAmount >= goal.targetAmount);

      setGoals(activeGoals);
      setPastGoals(expiredGoals);
    } catch (error) {
      console.error("Error fetching goals:", error);
    } finally {
      setLoading(false);
    }
  };

  const joinChallenge = async (goalId) => {
    try {
      await axios.put(`https://backend-johh.onrender.com/api/family-goals/${goalId}/join`, { userId });
      fetchGoals();
    } catch (error) {
      console.error("Error joining challenge:", error);
    }
  };

  const contributeAmount = async (goalId, amount) => {
    const contributionAmount = Number(amount);

    if (isNaN(contributionAmount) || contributionAmount <= 0) {
      alert("Please enter a valid amount!");
      return;
    }

    try {
      await axios.put(`https://backend-johh.onrender.com/api/family-goals/${goalId}/contribute`, {
        userId,
        amount: contributionAmount,
      });

      setContributions((prev) => ({ ...prev, [goalId]: "" }));
      fetchGoals();
    } catch (error) {
      console.error("Error contributing:", error);
    }
  };

  return (
    <div style={{ maxWidth: "600px", margin: "20px auto", padding: "20px", borderRadius: "10px", background: "#f8f9fa", boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)" }}>
      <h2 style={{ textAlign: "center", color: "#333" }}>Family Expense Challenges</h2>

      {loading ? (
        <p>Loading...</p>
      ) : (
        <>
          {!showPastGoals ? (
            <>
              {goals.length === 0 ? (
                <p style={{ textAlign: "center", color: "green" }}>🎉 All active goals are completed!</p>
              ) : (
                <ul style={{ listStyle: "none", padding: "0" }}>
                  {goals.map((goal) => (
                    <li key={goal._id} style={{ background: "#ffffff", padding: "15px", marginBottom: "10px", borderRadius: "8px", borderLeft: "5px solid #007bff", boxShadow: "2px 2px 6px rgba(0, 0, 0, 0.1)" }}>
                      <h3 style={{ fontSize: "18px", margin: "5px 0" }}>
                        {goal.goalName} - ₹{goal.currentAmount} / ₹{goal.targetAmount}
                      </h3>
                      <p style={{ fontSize: "14px", color: "#555" }}>Status: {goal.status}</p>
                      <p style={{ fontSize: "14px", color: "#555" }}>Reward: {goal.reward}</p>

                      {goal.participants.some((p) => p.userId === userId) ? (
                        <div style={{ marginTop: "10px" }}>
                          <input
                            type="number"
                            placeholder="Enter amount"
                            value={contributions[goal._id] || ""}
                            onChange={(e) => setContributions({ ...contributions, [goal._id]: e.target.value })}
                            style={{ padding: "8px", borderRadius: "5px", border: "1px solid #ccc", marginRight: "10px", width: "80px" }}
                          />
                          <button
                            style={{ background: "#28a745", color: "white", border: "none", padding: "8px 12px", borderRadius: "5px", cursor: "pointer", transition: "0.3s", fontSize: "14px" }}
                            onClick={() => contributeAmount(goal._id, contributions[goal._id])}
                            onMouseOver={(e) => (e.target.style.background = "#218838")}
                            onMouseOut={(e) => (e.target.style.background = "#28a745")}
                          >
                            Contribute
                          </button>
                        </div>
                      ) : (
                        <button
                          style={{ background: "#007bff", color: "white", border: "none", padding: "8px 12px", borderRadius: "5px", cursor: "pointer", transition: "0.3s", fontSize: "14px" }}
                          onClick={() => joinChallenge(goal._id)}
                          onMouseOver={(e) => (e.target.style.background = "#0056b3")}
                          onMouseOut={(e) => (e.target.style.background = "#007bff")}
                        >
                          Join Challenge
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              )}

              {/* View past goals button */}
              {pastGoals.length > 0 && (
                <button
                  style={{
                    marginTop: "15px",
                    background: "#6c757d",
                    color: "white",
                    border: "none",
                    padding: "10px 15px",
                    borderRadius: "5px",
                    cursor: "pointer",
                    display: "block",
                    width: "100%",
                  }}
                  onClick={() => setShowPastGoals(true)}
                >
                  View Past Goals & Savings
                </button>
              )}
            </>
          ) : (
            <>
              <h3 style={{ textAlign: "center", color: "#555" }}>Past Goals & Savings</h3>
              <ul style={{ listStyle: "none", padding: "0" }}>
                {pastGoals.map((goal) => (
                  <li key={goal._id} style={{ background: "#e9ecef", padding: "15px", marginBottom: "10px", borderRadius: "8px", borderLeft: "5px solid #6c757d", boxShadow: "2px 2px 6px rgba(0, 0, 0, 0.1)" }}>
                    <h3 style={{ fontSize: "18px", margin: "5px 0" }}>
                      {goal.goalName} - ₹{goal.currentAmount} / ₹{goal.targetAmount}
                    </h3>
                    <p style={{ fontSize: "14px", color: "#555" }}>Status: {goal.currentAmount >= goal.targetAmount ? "Completed" : "Expired"}</p>
                    <p style={{ fontSize: "14px", color: "#555" }}>Reward: {goal.reward}</p>
                  </li>
                ))}
              </ul>
              <button
                style={{
                  marginTop: "15px",
                  background: "#007bff",
                  color: "white",
                  border: "none",
                  padding: "10px 15px",
                  borderRadius: "5px",
                  cursor: "pointer",
                  display: "block",
                  width: "100%",
                }}
                onClick={() => setShowPastGoals(false)}
              >
                Back to Active Goals
              </button>
            </>
          )}
        </>
      )}
    </div>
  );
};

export default FamilyGoals;
