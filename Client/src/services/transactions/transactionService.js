import axios from "axios";
import { BASE_URL } from "../../utils/url";
import { getUserFromStorage } from "../../utils/getUserFromStorage";
const API_BASE_URL = "https://backend-johh.onrender.com/api/v1/transactions"; // Adjust your backend API URL if needed
//! Get the token
const token = getUserFromStorage();
//! Add
export const addTransactionAPI = async ({
  type,
  category,
  date,
  description,
  amount,
  splitExpense,  // ✅ Add this
  splitUsers,    // ✅ Add this
  user,   
}) => {

  const response = await axios.post(
    `${BASE_URL}/transactions/create`,
    {
      category,
      date,
      description,
      amount,
      type,
      splitExpense,  // ✅ Send to backend
      splitUsers,    // ✅ Send to backend
      user, 
    },
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );
  //Return a promise
  return response.data;
};
//! update
export const updateCategoryAPI = async ({ name, type, id }) => {
  const response = await axios.put(
    `${BASE_URL}/categories/update/${id}`,
    {
      name,
      type,
    },
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );
  //Return a promise
  return response.data;
};
//! delete
export const deleteCategoryAPI = async (id) => {
  const response = await axios.delete(`${BASE_URL}/categories/delete/${id}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  //Return a promise
  return response.data;
};
//! lists
export const listTransactionsAPI = async ({
  category,
  type,
  startDate,
  endDate,
}) => {
  const response = await axios.get(`${BASE_URL}/transactions/lists`, {
    params: { category, endDate, startDate, type },
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  //Return a promise
  return response.data;
};


// Delete a transaction by ID
export const deleteTransactionAPI = async (transactionId) => {

  try {
    console.log(transactionId);
    const response = await axios.delete(`${API_BASE_URL}/delete/${transactionId}`,
      {headers: {
      Authorization: `Bearer ${token}`,
    },});
    console.log(response);
    return response.data;
  } catch (error) {
    console.error("Error deleting transaction:", error);
    throw error;
  }
};

// Update a transaction by ID
export const updateTransactionAPI = async (transactionId, updatedData) => {
  try {
    const response = await axios.put(`${API_BASE_URL}/update/${transactionId}`, updatedData,
      {headers: {
      Authorization: `Bearer ${token}`,
    },});
    return response.data;
  } catch (error) {
    console.error("Error updating transaction:", error);
    throw error;
  }
};

