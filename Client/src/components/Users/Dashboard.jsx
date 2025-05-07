import React, { useEffect, useState } from "react";
import { useDispatch,useSelector } from "react-redux";
import axios from "axios";
import TransactionChart from "../Transactions/TransactionChart";
import TransactionList from "../Transactions/TransactionList";
import TaxReport from "../Reports/TaxReport";
import Budget from "../Transactions/Budget";
import { fetchBudget } from "../../redux/slice/budgetSlice";
import { Pie, Bar } from "react-chartjs-2";
import { addDebt,fetchDebt, payEMI, deleteDebt } from "../../redux/slice/debtSlice";
import { useNavigate } from "react-router-dom";
// import VoiceAssistant from "../VoiceAssistant/VoiceAssistant"; 
import FamilyGoals from "../FamilyGoals/FamilyGoals";
import {deleteSubscription, enableNotification,addSubscription, getSubscriptions,disableNotification } from "../../services/subscriptionService/subscriptionService";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
} from "chart.js";
import {BellIcon, BellSlashIcon, TrashIcon } from "@heroicons/react/24/outline";

ChartJS.register(ArcElement, Tooltip, Legend, BarElement, CategoryScale, LinearScale);
const Dashboard = () => 
  {
    const token = useSelector((state) => state.auth.token); // 🔥 Get token from Redux
    const dispatch = useDispatch();
    const [subscriptions, setSubscriptions] = useState([]);
  const user = useSelector((state) => state.auth.user);
  const [leftoverAmount, setLeftoverAmount] = useState(null);
  const [allocated, setAllocated] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newBudget, setNewBudget] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [updateStatus, setUpdateStatus] = useState("");
  const debts = useSelector((state) => state.debt.debts); // Fetching debts from Redux
  const [showAddDebt, setShowAddDebt] = useState(false);
  const [newDebtAmount, setNewDebtAmount] = useState("");
  const [InterestRate, setInterestRate] = useState("");
  const [DebtEMI, setDebtEMI] = useState("");
  const[amount,setAmount]=useState("");
  const[loanPurpose,setloanPurpose]=useState("");
  const currentMonth = new Date().toISOString().substring(0, 7);
  const [showForm, setShowForm] = useState(false); // Toggle form visibility
  const [subscriptionData, setSubscriptionData] = useState({
    serviceName: "",
    amount: "",
    renewalDate: "",
    frequency: "monthly",
    alertDaysBefore: 3,
  });

  useEffect(() => {
    const fetchSubscriptions = async () => {
      if (!token || !user?.id) return; // Ensure both are available
  
      try {
        const { data } = await getSubscriptions(token, user.id); // Pass both token and userId
        setSubscriptions(data);
      } catch (error) {
        console.error("Error fetching subscriptions:", error);
      }
    };
  
    fetchSubscriptions();
  }, [token, user?.id]); // Depend on both token and user ID
  

  const handleEnable = async (subscriptionId) => {
    if (!token) return alert("Please log in first.");
    try {
      await enableNotification(token, subscriptionId);
      alert("Subscription enabled successfully!");
      setSubscriptions(subscriptions.map(sub => 
        sub._id === subscriptionId ? { ...sub, notify: true } : sub
      ));
    } catch (error) {
      console.error("Error enabling subscription:", error);
      alert("Failed to enable subscription.");
    }
  };
  
  const handleDisable = async (subscriptionId) => {
    if (!token) return alert("Please log in first.");
    try {
      await disableNotification(token, subscriptionId);
      alert("Subscription disabled successfully!");
      setSubscriptions(subscriptions.map(sub => 
        sub._id === subscriptionId ? { ...sub, notify: false } : sub
      ));
    } catch (error) {
      console.error("Error disabling subscription:", error);
      alert("Failed to disable subscription.");
    }
  };
  
  const handleDelete = async (subscriptionId) => {
    if (!token) return alert("Please log in first.");
    try {
      await deleteSubscription(token, subscriptionId);
      alert("Subscription deleted successfully!");
      setSubscriptions(subscriptions.filter(sub => sub._id !== subscriptionId));
    } catch (error) {
      console.error("Error deleting subscription:", error);
      alert("Failed to delete subscription.");
    }
  };
  
  

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSubscriptionData({ ...subscriptionData, [name]: value });
  };

  const handleAddSubscription = async () => {
    if (!token) return alert("Please log in first.");
     try {
      const { data } = await addSubscription(token, subscriptionData);    
      alert("Subscription added!");
      setSubscriptions([...subscriptions,data]); // Update UI
      setShowForm(false); // Hide form after submission
    } catch (error) {
      console.error("Error adding subscription:", error);
    }
  };

  useEffect(() => {
    if (user?.id && debts.length === 0) {
      dispatch(fetchDebt());
    }
  }, [dispatch, user?.id]); // Removed `debts.length` to avoid unnecessary re-fetch
  

  const handlePayEMI = async (debtId) => {
    const amount = prompt("Enter EMI amount:");
    if (amount && !isNaN(amount) && amount > 0) {
      try {
         await dispatch(payEMI({ debtId, amount })).unwrap(); 
         dispatch(fetchDebt()); // Immediately refetch debts
      } catch (error) {
        alert("Failed to process EMI payment.");
      }
    } else {
      alert("Invalid amount entered.");
    }
  };
  
  
  // const handleDeleteDebt = async(debtId) => {
  //   await  dispatch(deleteDebt(debtId));
  //   setTimeout(() => dispatch(fetchDebt()), 500); // Fetch updated debts after a short delay
  // };


  const handleDeleteDebt = async (debtId) => {
    try {
      await dispatch(deleteDebt(debtId)).unwrap();
      dispatch(fetchDebt()); // Immediately refetch debts
    } catch (error) {
      alert("Failed to delete debt.");
    }
  };
  

  const handleAddDebt = () => {
    if (!amount || !InterestRate || !DebtEMI||!loanPurpose) {
      alert("Please enter all fields.");
      return;
    }
  
    dispatch(
      addDebt({
        amount: parseFloat(amount),
        loanPurpose,
        interestRate: parseFloat(InterestRate),
        monthlyEMI: parseFloat(DebtEMI),
      })
    );
  
    setAmount("");
    setInterestRate("");
    setDebtEMI("");
    setShowAddDebt(false);
  };
  
  useEffect(() => {
    const fetchBudget = async () => {
      try {
        if (user && user.id) {
          const response = await axios.get(
            `https://backend-johh.onrender.com/api/budgets/lists?month=${currentMonth}`,
            { headers: { Authorization: `Bearer ${user.token}` } }
          );
          setLeftoverAmount(response.data.leftoverAmount);
          setAllocated(response.data.allocatedBudget);
        }
      } catch (err) {
        setError("Failed to fetch budget.");
      } finally {
        setLoading(false);
      }
    };

    fetchBudget();
  }, [user, currentMonth]);


   
  
  const updateBudget = async () => {
    if (!newBudget || isNaN(newBudget) || newBudget <= 0) {
        setUpdateStatus("Please enter a valid amount.");
        return;
    }

    try {
        const response = await axios.put(
            `https://backend-johh.onrender.com/api/budgets/update`,
            { amount: newBudget, month: currentMonth },
            { headers: { Authorization: `Bearer ${user.token}` } }
        );

        setAllocated(response.data.allocatedBudget);
        setLeftoverAmount(response.data.leftoverAmount); // Set correct leftover amount
        setUpdateStatus("Budget updated successfully!");
       
        setShowModal(false);
      
    } catch (error) {
        setUpdateStatus("Failed to update budget. Try again.");
    }
};
        const [familyExpenses, setFamilyExpenses] = useState([]);

        useEffect(() => {
          const fetchFamilyExpenses = async () => {
            try {
              const response = await axios.get("https://backend-johh.onrender.com/family/family-expenses", {
                headers: { Authorization: `Bearer ${user.token}` },
              });
              setFamilyExpenses(response.data);
            } catch (error) {
              console.error("Failed to fetch family expenses:", error);
            }
          };

          fetchFamilyExpenses();
        }, [user]);

// Prepare chart data
        const chartData = {
          labels: familyExpenses.map((expense) => expense._id),
          datasets: [
            {
              data: familyExpenses.map((expense) => expense.totalAmount),
              backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"],
            },
          ],
        };


  return (
    <>
      <TransactionChart />

       <div className="dashboard">
      <h2 className="text-center text-2xl font-bold mb-4">Family Expense Overview</h2>
      
     
              <div className="flex flex-col items-center gap-8 p-4">
          {/* Pie Chart */}
          <div className="w-full max-w-lg h-[400px] bg-white p-4 rounded-lg shadow-md">
            <Pie data={chartData} options={{ maintainAspectRatio: false }} />
          </div>

          {/* Bar Chart */}
          <div className="w-full max-w-2xl h-[400px] bg-white p-4 rounded-lg shadow-md">
            <Bar
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } },
              }}
            />
          </div>
        </div>

            </div>
      <TransactionList />
        
  {/* Family Expense Challenges Section */}
  {user?.familyId && <FamilyGoals familyId={user.familyId} userId={user.id} />}
   
      {/* Show Leftover Budget only if budget is set */}
      {allocated !== null && allocated !== undefined ? (
        <div className="budget-info">
          {loading ? (
            <p>Loading budget...</p>
          ) : (
            <div className="max-w-md mx-auto mt-6 p-4 bg-gradient-to-r from-green-400 to-blue-500 text-white rounded-lg shadow-md text-center">
              <h2 className="text-2xl font-bold">Monthly Budget</h2>
              <p className="text-lg mt-2">
              {leftoverAmount < 0 ? (
                <span className="font-semibold text-xl text-red-500">
                  ⚠️ You have exceeded your budget by ₹{Math.abs(leftoverAmount)}
                </span>
              ) : (
                <>
                  💰 <span className="font-semibold text-xl">₹{leftoverAmount}</span>
                  <p className="text-sm mt-1 opacity-80">remaining to spend this month</p>
                </>
              )}
            </p>


              {/* Update Budget Button */}
              <button
                className="mt-3 px-4 py-1 text-sm font-medium bg-white text-blue-600 rounded-full shadow hover:bg-gray-100 transition duration-200"
                onClick={() => setShowModal(true)}
              >
                Update Budget
              </button>
            </div>
          )}
        </div>
      ) : null}


      {/* Update Budget Modal */}
      {showModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-5 rounded-lg shadow-lg text-center w-96">
            <h3 className="text-xl font-semibold">Update Budget</h3>
            <input
              type="number"
              value={newBudget}
              onChange={(e) => setNewBudget(e.target.value)}
              className="border mt-3 p-2 w-full rounded-md"
              placeholder="Enter new budget"
            />
            {updateStatus && <p className="text-sm mt-2 text-red-500">{updateStatus}</p>}
            <div className="mt-4 flex justify-between">
              <button
                className="px-4 py-2 bg-gray-300 rounded-md"
                onClick={() => setShowModal(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                onClick={updateBudget}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
           
              {/* Debt Section */}
      <div className="debt-section bg-white p-4 rounded-lg shadow-md">
        <h3 className="text-xl font-semibold mb-3">Debt Overview</h3>

        {debts.length === 0 ? (
          <div className="text-center p-4">
            <p className="text-gray-600">Do you have any debt?</p>
            <button
              onClick={() => setShowAddDebt(true)}
              className="mt-2 px-4 py-2 bg-blue-500 text-white rounded-md"
            >
              Add Debt
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {debts
            .filter((debt) => debt.status !== "Paid") 
            .map((debt) => (
              <div key={debt._id} className="p-4 border rounded-lg flex justify-between items-center">
                <div>
                <p><strong>Left Amount:</strong> ₹{debt.remainingAmount.toFixed(2)}</p>
                <p><strong>Purpose:</strong> ₹{debt.loanPurpose}</p>
                <p><strong>Interest Rate:</strong> {debt.interestRate}%</p>
                <p><strong>Monthly EMI:</strong> ₹{debt.monthlyEMI}</p>
                </div>
                <div className="space-x-2">
                  
                  <button
                  
                    onClick={() => handleDeleteDebt(debt._id)}
                    className="px-4 py-2 bg-red-500 text-white rounded-md"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}

            {/* + Add More Debt Button */}
            <button
              onClick={() => setShowAddDebt(true)}
              className="mt-2 px-4 py-2 bg-blue-500 text-white rounded-md"
            >
              + Add More
            </button>
          </div>
        )}
      </div>

{subscriptions.length === 0 ? (
  <div className="bg-gray-50 p-6 rounded-lg shadow-md text-center border border-gray-200">
    <p className="text-gray-700 font-medium text-lg">
      Stay updated with daily insights! Add a subscription now.
    </p>
    <button 
      onClick={() => setShowForm(true)} 
      className="mt-4 bg-blue-500 text-white font-semibold px-5 py-2 rounded-lg shadow-md hover:bg-blue-600 transition duration-300">
      Subscribe
    </button>
  </div>
) : (
  <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-200">
    <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
      Your Subscriptions
    </h3>
  
<ul className="bg-white p-4 rounded-md shadow-md">
  {subscriptions.map((sub, index) => (
    <li key={index} className="border-b last:border-none py-2 text-gray-700 flex justify-between items-center">
      <span>
        {sub.serviceName} - <span className="font-semibold">${sub.amount}</span> - {sub.frequency}
      </span>
      <div className="flex space-x-2">
        {sub.notify ? (
          <button 
            onClick={() => handleDisable(sub._id)} 
            className="p-2 text-red-500 hover:text-red-700 transition"
          >
            <BellSlashIcon className="w-6 h-6" />
          </button>
        ) : (
          <button 
            onClick={() => handleEnable(sub._id)} 
            className="p-2 text-green-500 hover:text-green-700 transition"
          >
            <BellIcon className="w-6 h-6" />
          </button>
        )}
        <button 
          onClick={() => handleDelete(sub._id)} 
          className="p-2 text-gray-500 hover:text-gray-700 transition"
        >
          <TrashIcon className="w-6 h-6" />
        </button>
      </div>
    </li>
  ))}
</ul>


    <button 
      onClick={() => setShowForm(true)} 
      className="mt-5 w-full bg-green-500 text-white font-semibold py-2 rounded-lg shadow-md hover:bg-green-600 transition duration-300">
      + Add Subscription
    </button>
  </div>
)}

{showForm && (
  <div className="bg-white p-6 rounded-lg shadow-lg mt-4">
    <h3 className="text-lg font-semibold text-gray-800 mb-4">Add Subscription</h3>

    <label className="block font-medium mb-1">Service Name:</label>
    <input 
      type="text" 
      name="serviceName" 
      value={subscriptionData.serviceName} 
      onChange={handleInputChange} 
      required 
      className="w-full p-2 border rounded-md mb-3"
    />

    <label className="block font-medium mb-1">Amount:</label>
    <input 
      type="number" 
      name="amount" 
      value={subscriptionData.amount} 
      onChange={handleInputChange} 
      required 
      className="w-full p-2 border rounded-md mb-3"
    />

    <label className="block font-medium mb-1">Renewal Date:</label>
    <input 
      type="date" 
      name="renewalDate" 
      value={subscriptionData.renewalDate} 
      onChange={handleInputChange} 
      required 
      className="w-full p-2 border rounded-md mb-3"
    />

    <label className="block font-medium mb-1">Frequency:</label>
    <select 
      name="frequency" 
      value={subscriptionData.frequency} 
      onChange={handleInputChange} 
      className="w-full p-2 border rounded-md mb-3"
    >
      <option value="weekly">Weekly</option>
      <option value="monthly">Monthly</option>
      <option value="yearly">Yearly</option>
    </select>

    <label className="block font-medium mb-1">Alert Days Before Renewal:</label>
    <input 
      type="number" 
      name="alertDaysBefore" 
      value={subscriptionData.alertDaysBefore} 
      onChange={handleInputChange} 
      className="w-full p-2 border rounded-md mb-3"
    />

    <div className="flex gap-3 mt-4">
      <button 
        onClick={handleAddSubscription} 
        className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition">
        Save Subscription
      </button>
      <button 
        onClick={() => setShowForm(false)} 
        className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600 transition">
        Cancel
      </button>
    </div>
  </div>
)}

      {/* Add Debt Modal */}
      {showAddDebt && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <div className="bg-white p-5 rounded-lg shadow-lg text-center w-96">
                <h3 className="text-xl font-semibold">Add Debt</h3>

                <input
                  type="number"
                  placeholder="Amount"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  className="border mt-2 p-2 w-full rounded-md"
                />
                <input
                  type="text"
                  placeholder="Loan Purpose"
                  value={loanPurpose}
                  onChange={(e) => setloanPurpose(e.target.value)}
                  className="border mt-2 p-2 w-full rounded-md"
                />
                <input
                  type="number"
                  placeholder="Interest Rate (%)"
                  value={InterestRate}
                  onChange={(e) => setInterestRate(e.target.value)}
                  className="border mt-2 p-2 w-full rounded-md"
                />
                <input
                  type="number"
                  placeholder="Monthly EMI"
                  value={DebtEMI}
                  onChange={(e) => setDebtEMI(e.target.value)}
                  className="border mt-2 p-2 w-full rounded-md"
                />

                <div className="mt-4 flex justify-between">
                  <button
                    className="px-4 py-2 bg-gray-300 rounded-md"
                    onClick={() => setShowAddDebt(false)}
                  >
                    Cancel
                  </button>
                  <button
                    className="px-4 py-2 bg-blue-500 text-white rounded-md"
                    onClick={handleAddDebt}
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          )}

 
            {/* <VoiceAssistant /> */}
      {/* Show Tax Report only if user exists */}
      {user && user.id ? (
        <TaxReport userId={user.id} />
      ) : (
        <div className="flex items-center">
          <p>Loading tax report...</p>
          <span className="loader ml-2"></span>
        </div>
      )}
    </>
  );
};

export default Dashboard;
