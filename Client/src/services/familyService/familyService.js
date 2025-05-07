import axios from "axios";

const API_URL = "https://backend-johh.onrender.com";

export const createFamily = async (familyName, adminId) => {
    return axios.post(`${API_URL}/create`, { familyName, adminId });
};

export const addMember = async (familyId, adminId, userId) => {
    return axios.post(`${API_URL}/add-member`, { familyId, adminId, userId });
};

export const removeMember = async (familyId, adminId, userId) => {
    return axios.post(`${API_URL}/remove-member`, { familyId, adminId, userId });
};

export const assignRole = async (familyId, adminId, userId, role) => {
    return axios.post(`${API_URL}/assign-role`, { familyId, adminId, userId, role });
};



// export const getFamilyDetails = async (familyId) => {
//     try {
//         const response = await axios.get(`${API_URL}/${familyId}`);
       
//         return response.data.family;
//     } catch (error) {
//         console.error("Error fetching family details:", error);
//         return null;
//     }
// };

export const getFamilyDetails = async (familyId) => {
  try {
      console.log(`Fetching family details for ID: ${familyId}`); // Debugging  
      const response = await axios.get(`https://backend-johh.onrender.com/${familyId}`);
      console.log("API Response:", response.data); // Debugging  
      return response.data;  
  } catch (error) {
      console.error("Error fetching family details:", error);
      //return null; // Return null if error occurs  
  }
};

export const listFamilyMembersAPI = async (userId) => {
    try {
        console.log(userId);
      const response = await axios.get(`${API_URL}/getFamilyByUserId/${userId}`, {
        // params: { userId },
      });
    //   console.log("response",response.data);
      return response.data;
    } catch (error) {
      console.error("Error fetching family members:", error);
      throw error;
    }
  };