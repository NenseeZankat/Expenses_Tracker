import axios from "axios";

const API_URL = "https://backend-johh.onrender.com/api/subscriptions"; // Adjust if needed

export const addSubscription = async (token, subscriptionData) => {
  return axios.post(`${API_URL}/add`, subscriptionData, {
    headers: { Authorization: `Bearer ${token}` }
  });
};

export const getSubscriptions = async (token, userId) => {
  return axios.get(`${API_URL}/${userId}`, {
    headers: { Authorization: `Bearer ${token}` }
  });
};

export const disableNotification = async (token, subscriptionId) => {
  return axios.put(`${API_URL}/${subscriptionId}/disable`, {}, {
    headers: { Authorization: `Bearer ${token}` }
  });
};

export const enableNotification = async (token, subscriptionId) => {
  return axios.put(`${API_URL}/${subscriptionId}/enable`, {}, {
    headers: { Authorization: `Bearer ${token}` },
  });
};

// 🗑️ Delete Subscription
export const deleteSubscription = async (token, subscriptionId) => {
  return axios.delete(`${API_URL}/${subscriptionId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
};