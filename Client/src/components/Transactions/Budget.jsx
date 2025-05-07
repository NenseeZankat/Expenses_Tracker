
// import React, { useState, useEffect } from "react";
// import { useDispatch, useSelector } from "react-redux";
// import { fetchBudget, setBudget } from "../../redux/slice/budgetSlice";

// const Budget = () => {
//   const [month, setMonth] = useState("");
//   const [allocatedBudget, setAllocatedBudget] = useState("");
//   const dispatch = useDispatch();
//   const { budget, loading, error } = useSelector((state) => state.budget);
//   const [budgetExistsError, setBudgetExistsError] = useState("");
//   const [localBudget, setLocalBudget] = useState(null);

//   useEffect(() => {
//     if (month) {
//       console.log("Fetching budget for:", month);
//       dispatch(fetchBudget(month));
//     }
//   }, [month, dispatch]);

//   const handleSetBudget = async () => {
//     if (month && allocatedBudget) {
//       try {
//         setBudgetExistsError("");

//         const action = await dispatch(setBudget({ month, allocatedBudget }));

//         if (setBudget.rejected.match(action)) {
//           const errorMessage = action.payload?.error || "An error occurred.";
//           setBudgetExistsError(errorMessage);
//         } else {
//           const newLeftover = allocatedBudget - (budget?.totalExpenses || 0);

//           setLocalBudget({
//             allocatedBudget: allocatedBudget,
//             totalExpenses: budget?.totalExpenses || 0,
//             leftoverAmount: newLeftover,
//           });
//         }
//       } catch (error) {
//         setBudgetExistsError(error.message || "An error occurred while setting the budget.");
//       }
//     }
//   };

//   return (
//     <div className="p-4">
//       <h2 className="text-xl font-bold">Budget Management</h2>

//       <input
//         type="month"
//         value={month}
//         onChange={(e) => setMonth(e.target.value)}
//         className="border p-2 rounded w-full"
//       />

//       <input
//         type="number"
//         placeholder="Set Budget Amount"
//         value={allocatedBudget}
//         onChange={(e) => setAllocatedBudget(e.target.value)}
//         className="border p-2 rounded w-full mt-2"
//       />

//       <button onClick={handleSetBudget} className="bg-blue-500 text-white p-2 rounded mt-2">
//         Save Budget
//       </button>

//       {loading && <p>Loading...</p>}
//       {error && <p className="text-red-500">{error}</p>}
//       {budgetExistsError && <p className="text-red-500">{budgetExistsError}</p>}

//       {!budgetExistsError && localBudget && (
//         <div className="mt-4">
//           <h3 className="text-lg font-semibold">Current Budget: ${localBudget.allocatedBudget}</h3>
//           <p>Total Expenses: ${localBudget.totalExpenses}</p>
//           <p>Leftover Amount: ${localBudget.leftoverAmount}</p>

//           {localBudget.leftoverAmount < 0 && (
//             <p className="text-red-500 font-semibold">
//               Warning: Expenses exceeded the budget!
//             </p>
//           )}
//         </div>
//       )}
//     </div>
//   );
// };

// export default Budget;


import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { fetchBudget, setBudget, updateBudget } from "../../redux/slice/budgetSlice";

const Budget = () => {
  const [month, setMonth] = useState("");
  const [allocatedBudget, setAllocatedBudget] = useState("");
  const dispatch = useDispatch();
  const { budget, loading, error } = useSelector((state) => state.budget);
  const [budgetExistsError, setBudgetExistsError] = useState("");
  const [localBudget, setLocalBudget] = useState(null);
  const [isUpdating, setIsUpdating] = useState(false); // Track if updating budget
  const [amount, setamountt] = useState(null);
 
  useEffect(() => {
    if (month) {
      console.log("Fetching budget for:", month);
      dispatch(fetchBudget(month));
      setIsUpdating(false); // Reset update mode when month changes
      setBudgetExistsError(""); // Clear previous errors
      setLocalBudget(null); // Clear displayed budget details
    }
  }, [month, dispatch]);
  
  const handleSetBudget = async () => {
    if (month && allocatedBudget) {
      try {
        setBudgetExistsError("");

        const action = await dispatch(setBudget({ month, allocatedBudget }));

        if (setBudget.rejected.match(action)) {
          const errorMessage = action.payload?.error || "An error occurred.";
          setBudgetExistsError(errorMessage);

          if (errorMessage.includes("already exists")) {
            setIsUpdating(true); // Show "Update Budget" button if budget exists
          }
        } else {
          setIsUpdating(false); // Hide update button if budget is set successfully

          const newLeftover = allocatedBudget - (budget?.totalExpenses || 0);

          setLocalBudget({
            allocatedBudget: allocatedBudget,
            totalExpenses: budget?.totalExpenses || 0,
            leftoverAmount: newLeftover,
          });
        }
      } catch (error) {
        setBudgetExistsError(error.message || "An error occurred while setting the budget.");
      }
    }
  };

  // const handleUpdateBudget = () => {
  //   setIsUpdating(false);
  //   handleSetBudget(); // Call the same function to update the budget
  // };

  // const handleUpdateBudget = async () => {
  //   if (!month || !allocatedBudget) return;

  //   try {
  //     setBudgetExistsError("");

  //     const action = await dispatch(updateBudget({ month, allocatedBudget }));

  //     if (updateBudget.rejected.match(action)) {
  //       setBudgetExistsError(action.payload?.error || "An error occurred.");
  //     }
  //   } catch (error) {
  //     setBudgetExistsError(error.message || "An error occurred while updating the budget.");
  //   }
  // };




  const handleUpdateBudget = async () => {
    if (!month || !allocatedBudget) {
      setBudgetExistsError("Month and new budget amount are required.");
      return;
    }
  
    console.log("Updating budget with:", { month, allocatedBudget });
    setamountt(allocatedBudget);
    try {
      setBudgetExistsError("");
      const action = await dispatch(updateBudget({ month,allocatedBudget}));
  
      if (updateBudget.rejected.match(action)) {
        setBudgetExistsError(action.payload?.error || "An error occurred.");
      } else {
        console.log("Budget updated successfully:", action.payload);
      }
    } catch (error) {
      setBudgetExistsError(error.message || "An error occurred while updating the budget.");
    }
  };
  

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold">Budget Management</h2>

      <input
        type="month"
        value={month}
        onChange={(e) => setMonth(e.target.value)}
        className="border p-2 rounded w-full"
      />

      <input
        type="number"
        placeholder="Set Budget Amount"
        value={allocatedBudget}
        onChange={(e) => setAllocatedBudget(e.target.value)}
        className="border p-2 rounded w-full mt-2"
      />

      {!isUpdating ? (
        <button onClick={handleSetBudget} className="bg-blue-500 text-white p-2 rounded mt-2">
          Save Budget
        </button>
      ) : (
        <button onClick={handleUpdateBudget} className="bg-yellow-500 text-white p-2 rounded mt-2">
          Update Budget
        </button>
      )}

      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">{error}</p>}
      {budgetExistsError && <p className="text-red-500">{budgetExistsError}</p>}

      {!budgetExistsError && localBudget && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold">Current Budget: ${localBudget.allocatedBudget}</h3>
          <p>Total Expenses: ${localBudget.totalExpenses}</p>
          <p>Leftover Amount: ${localBudget.leftoverAmount}</p>

          {localBudget.leftoverAmount < 0 && (
            <p className="text-red-500 font-semibold">
              Warning: Expenses exceeded the budget!
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default Budget;
