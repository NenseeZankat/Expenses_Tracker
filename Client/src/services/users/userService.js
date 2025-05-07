import axios from "axios";
import { BASE_URL } from "../../utils/url";
import { getUserFromStorage } from "../../utils/getUserFromStorage";
//! Get the token
const token = getUserFromStorage();
//! Login
export const loginAPI = async ({ email, password }) => {
  console.log(email);
  const response = await axios.post(`${BASE_URL}/users/login`, {
    email,
    password,
  });
  //Return a promise
  return response.data;
};
//! register
export const registerAPI = async ({ email, password, username }) => {

  const response = await axios.post(`${BASE_URL}/users/register`, {
    email,
    password,
    username,
  });
  //Return a promise
  return response.data;
};
//! change password
export const changePasswordAPI = async (newPassword) => {
  const response = await axios.put(
    `${BASE_URL}/users/change-password`,
    {
      newPassword,
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




//! update Profile
export const updateProfileAPI = async ({ username }) => {
  
  const response = await axios.put(
    `${BASE_URL}/users/update-profile`,
    {
      username,
    },
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );
  //Return a promise
  //console.log("hi"+username);
  return response.data;
};
export const getUserPreferenceAPI = async (userId) => {
  const response = await axios.get(`${BASE_URL}/users/${userId}/preference`, {
    headers,
  });
  return response.data;
};

//! Update User's Budget Allocation Preference
export const updateUserPreferenceAPI = async (userId, allocationPreference) => {
  const response = await axios.put(
    `${BASE_URL}/${userId}/allocation-preference`,
    { allocationPreference },
    { headers }
  );
  
  return response.data;
};
export const getUserDetailsAPI = async (token) => {
  const res = await fetch(`${BASE_URL}/users/profile`, {

    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!res.ok) throw new Error("Failed to fetch user data");
  console.log(res);
  return await res.json();
};

// export const updateProfilePicAPI = async (formData) => {
//   const response = await axios.put("http://localhost:8000/upload-profile-pic", formData);
//   return response.data;
// };

export const updateProfilePicAPI = async (formData, token) => {
  try {
    const response = await axios.put("https://backend-johh.onrender.com/upload-profile-pic", formData, {
      headers: {
        "Content-Type": "multipart/form-data", // Ensure this header for file upload
        Authorization: `Bearer ${token}`,     // Pass the token in the Authorization header
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading profile picture:", error.response?.data || error.message);
    throw error;
  }
};



// //! Get User's Budget Allocation Preference
// export const getUserPreferenceAPI = async (userId) => {
//   const response = await axios.get(`${BASE_URL}/users/${userId}/preference`, {
//     headers,
//   });
//   return response.data;
// };

// //! Update User's Budget Allocation Preference
// export const updateUserPreferenceAPI = async (userId, allocationPreference) => {
//   const response = await axios.put(
//     `${BASE_URL}/users/${userId}/preference`,
//     { allocationPreference },
//     { headers }
//   );
//   return response.data;
// };
