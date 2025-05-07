import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import {
  getSubscriptions,
  disableNotification,
  enableNotification,
  deleteSubscription,
  addSubscription,
} from "../../services/subscriptionService/subscriptionService";
import {
  BellIcon,
  BellSlashIcon,
  TrashIcon,
  PlusCircleIcon,
} from "@heroicons/react/24/outline";

const ManageSubscriptions = () => {
  const token = useSelector((state) => state.auth.token);
  const userId = useSelector((state) => state.auth.user?.id);
  const [subscriptions, setSubscriptions] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [subscriptionData, setSubscriptionData] = useState({
    serviceName: "",
    amount:"",
    renewalDate: "",
    frequency: "",
    alertDaysBefore: "",
  });
 
  useEffect(() => {
    const fetchSubscriptions = async () => {
      if (!token || !userId) return;
      try {
        const { data } = await getSubscriptions(token, userId);
        setSubscriptions(data);
      } catch (error) {
        console.error("Error fetching subscriptions:", error);
      }
    };

    fetchSubscriptions();
  }, [token, userId]);

  // Handle Form Input
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSubscriptionData({
      ...subscriptionData,
      [name]: name === "amount" || name === "alertDaysBefore" ? Number(value) :
      name === "renewalDate" ? new Date(value).toISOString() :  value,
    });
  };
  
  // Add Subscription
  const handleAddSubscription = async () => {
    if (!token) return alert("Please log in first.");
    if (!subscriptionData.serviceName || !subscriptionData.amount || !subscriptionData.frequency || !subscriptionData.renewalDate)
      return alert("Please fill in all required fields.");
    const formattedData = {
      ...subscriptionData,
      amount: Number(subscriptionData.amount),
      alertDaysBefore: Number(subscriptionData.alertDaysBefore),
      renewalDate: new Date(subscriptionData.renewalDate).toISOString(), // Convert to Date Format
    };
    try {
      console.log("Subscription Data:", subscriptionData);
      const { data } = await addSubscription(token, subscriptionData);

      alert("Subscription added!");
      setSubscriptions((prev) => [...prev, data.subscription]); // ✅ Update UI instantly
      setShowForm(false); // ✅ Hide form after submission
      setSubscriptionData({ serviceName: "", amount:"", frequency: "", renewalDate: "", alertDaysBefore:"" }); // ✅ Clear form fields
    } catch (error) {
      console.error("Error adding subscription:", error);
      alert("Failed to add subscription.");
    }
  };


  const handleEnable = async (subscriptionId) => {
    if (!token) return alert("Please log in first.");
    try {
      await enableNotification(token, subscriptionId);
      alert("Subscription enabled successfully!");
      setSubscriptions((subs) =>
        subs.map((sub) =>
          sub._id === subscriptionId ? { ...sub, notify: true } : sub
        )
      );
    } catch (error) {
      console.error("Error enabling subscription:", error);
      alert("Failed to enable subscription.");
    }
  };

  // Disable Notifications
  const handleDisable = async (subscriptionId) => {
    if (!token) return alert("Please log in first.");
    try {
      await disableNotification(token, subscriptionId);
      alert("Subscription disabled successfully!");
      setSubscriptions((subs) =>
        subs.map((sub) =>
          sub._id === subscriptionId ? { ...sub, notify: false } : sub
        )
      );
    } catch (error) {
      console.error("Error disabling subscription:", error);
      alert("Failed to disable subscription.");
    }
  };

  // Delete Subscription
  const handleDelete = async (subscriptionId) => {
    if (!token) return alert("Please log in first.");
    if (!window.confirm("Are you sure you want to delete this subscription?"))
      return;
    try {
      await deleteSubscription(token, subscriptionId);
      alert("Subscription deleted successfully!");
      setSubscriptions((prev) =>
        prev.filter((sub) => sub._id !== subscriptionId)
      );
    } catch (error) {
      console.error("Error deleting subscription:", error);
    }
  };
  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-lg rounded-md">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">Manage Subscriptions</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition"
        >
          <PlusCircleIcon className="w-5 h-5 mr-2" /> Add Subscription
        </button>
      </div>

      {/* Add Subscription Form */}
      {showForm && (
        <div className="mb-4 p-4 bg-gray-100 rounded-md">
          <input
            type="text"
            name="serviceName"
            value={subscriptionData.serviceName}
            onChange={handleInputChange}
            placeholder="Service Name"
            className="p-2 border border-gray-300 rounded w-full mb-2"
          />
          <input
            type="number"
            name="amount"
            value={subscriptionData.amount}
            onChange={handleInputChange}
            placeholder="Amount"
            className="p-2 border border-gray-300 rounded w-full mb-2"
          />

          
          {/* Renewal Date Picker */}
          <input
            type="date"
            name="renewalDate"
            value={subscriptionData.renewalDate.split("T")[0]} // ✅ Display only YYYY-MM-DD
            onChange={handleInputChange}
            className="p-2 border border-gray-300 rounded w-full mb-2"
          />
          {/* Frequency Selection Dropdown */}
          <select
            name="frequency"
            value={subscriptionData.frequency}
            onChange={handleInputChange}
            className="p-2 border border-gray-300 rounded w-full mb-2"
          >
            <option value="">Select Frequency</option>
            <option value="Monthly">Monthly</option>
            <option value="Yearly">Yearly</option>
            <option value="weekly">weekly</option>

          </select>

          <input
            type="number"
            name="alertDaysBefore"
            value={subscriptionData.alertDaysBefore}
            onChange={handleInputChange}
            placeholder="Alert Days Before Renewal"
            className="p-2 border border-gray-300 rounded w-full mb-2"
          />
          <button
            onClick={handleAddSubscription}
            className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 transition w-full"
          >
            Save Subscription
          </button>
        </div>
      )}

      {/* Subscription List */}
      <table className="w-full border-collapse border border-gray-200">
        <thead>
          <tr className="bg-gray-100">
            <th className="border p-3 text-left">Plan</th>
            <th className="border p-3 text-left">Price</th>
            <th className="border p-3 text-left">Frequency</th>
            <th className="border p-3 text-left">Renewal Date</th>
            <th className="border p-3 text-center">Actions</th>
          </tr>
        </thead>
        <tbody>
          {subscriptions.length > 0 ? (
            subscriptions.map((sub) => (
              <tr key={sub._id} className="border">
                <td className="p-3">{sub.serviceName}</td>
                <td className="p-3">${sub.amount}</td>
                <td className="p-3">{sub.frequency}</td>
                <td className="p-3">{new Date(sub.renewalDate).toLocaleDateString()}</td> {/* ✅ Format Date */}                <td className="p-3 text-center flex justify-center gap-3">
                  {sub.notify ? (
                    <button
                      onClick={() => handleDisable(sub._id)}
                      className="p-2 rounded-full bg-gray-200 hover:bg-gray-300 transition"
                    >
                      <BellIcon className="w-5 h-5 text-green-500" />
                    </button>
                  ) : (
                    <button
                      onClick={() => handleEnable(sub._id)}
                      className="p-2 rounded-full bg-gray-200 hover:bg-gray-300 transition"
                    >
                      <BellSlashIcon className="w-5 h-5 text-red-500" />
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(sub._id)}
                    className="p-2 rounded-full bg-red-500 hover:bg-red-600 transition"
                  >
                    <TrashIcon className="w-5 h-5 text-white" />
                  </button>
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan="5" className="p-4 text-center text-gray-500">
                No subscriptions found. Click "Add Subscription" to get started.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default ManageSubscriptions;
