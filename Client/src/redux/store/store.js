import { configureStore } from "@reduxjs/toolkit";
import authReducer from "../slice/authSlice";
import debtReducer from "../slice/debtSlice";  // Ensure correct import
import budgetReducer from "../slice/budgetSlice";
import userReducer from '../slice/userSlice'; // Adjust path if necessary
export const store = configureStore({
  reducer: {
    auth: authReducer,
    budget: budgetReducer,
    debt: debtReducer,
    user: userReducer, 
  },
  
 
});
