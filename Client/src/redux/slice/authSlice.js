import { createSlice } from "@reduxjs/toolkit";

//!Initial State
const authSlice = createSlice({
  name: "auth",
  initialState: {
    user: JSON.parse(localStorage.getItem("userInfo")) || null,
    token: localStorage.getItem("token") || null, // Store token separately
  },
  //1 Reducers
  reducers: {
    loginAction: (state, action) => {
      state.user = action.payload;
      console.log("token",action.payload.token);
      console.log(action.payload);
      localStorage.setItem("userInfo", JSON.stringify(action.payload)); // Save user in localStorage
      localStorage.setItem("token", action.payload.token); // Save token in localStorage
    },

    updateUser: (state, action) => {
      state.user = { 
        ...state.user,  // Keep existing user data
        ...action.payload // Update with new data
      };

      // Update local storage
      localStorage.setItem("userInfo", JSON.stringify(action.payload));
    },
    //Logout
    logoutAction: (state, action) => {
      state.user = null;
      localStorage.removeItem("userInfo"); // Clear localStorage on logout
      localStorage.removeItem("token"); // Clear token on logout
    },
  },
});
//! Generate actions
export const { loginAction, logoutAction,updateUser } = authSlice.actions;
//! Generate the reducers
const authReducer = authSlice.reducer;
export default authReducer;













    
