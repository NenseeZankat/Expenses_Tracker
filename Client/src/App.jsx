import { BrowserRouter, Route, Routes } from "react-router-dom";
import HeroSection from "./components/Home/HomePage";
import PublicNavbar from "./components/Navbar/PublicNavbar";
import LoginForm from "./components/Users/Login";
import { useSelector } from "react-redux";
import "regenerator-runtime/runtime";
import { Elements } from "@stripe/react-stripe-js";

import ManageSubscriptions from "./components/Subscription/ManageSubscriptions"; // Import page
import RegistrationForm from "./components/Users/Register";
import PrivateNavbar from "./components/Navbar/PrivateNavbar";
import AddCategory from "./components/Category/AddCategory";
import CategoriesList from "./components/Category/CategoriesList";
import UpdateCategory from "./components/Category/UpdateCategory";
import TransactionForm from "./components/Transactions/TransactionForm";
import Dashboard from "./components/Users/Dashboard";
import UserProfile from "./components/Users/UserProfile";
import AuthRoute from "./components/Auth/AuthRoute";
import Budget from "./components/Transactions/Budget"; // Import Budget Page
import CheckoutForm from "./components/Transactions/CheckoutForm";
import Completion from "./components/Transactions/Completion";

import FamilyManager from "./components/Family/FamilyManager";
import CreateFamily from "./components/Family/CreateFamily";
import FamilyGoals from "./components/FamilyGoals/FamilyGoalForm";
import { loadStripe } from "@stripe/stripe-js";
const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY);

function App() {
  const stripeOptions = {
    mode: "payment",
    amount: 500,
    currency: "usd",
  };
  const user = useSelector((state) => state?.auth?.user);
  
  return (
    <BrowserRouter>
      {/* Navbar */}

      {user ? <PrivateNavbar /> : <PublicNavbar />}
      <Routes>
        <Route path="/" element={<HeroSection />} />
        <Route path="/login" element={<LoginForm />} />
        <Route path="/register" element={<RegistrationForm />} />
        <Route
          path="/add-category"
          element={
            <AuthRoute>
              <AddCategory />
            </AuthRoute>
          }
        />
        <Route
          path="/categories"
          element={
            <AuthRoute>
              <CategoriesList />
            </AuthRoute>
          }
        />
        <Route
          path="/completion"
          element={
            <AuthRoute>
                <Completion />
            </AuthRoute>
          }
        />
        <Route
          path="/update-category/:id"
          element={
            <AuthRoute>
              <UpdateCategory />
            </AuthRoute>
          }
        />
        <Route
          path="/add-purchase"
          element={
            <AuthRoute>
              <Elements stripe={stripePromise} options={stripeOptions}>
                <CheckoutForm />
              </Elements>
            </AuthRoute>
          }
        />
        <Route
          path="/add-transaction"
          element={
            <AuthRoute>
              <TransactionForm />
            </AuthRoute>
          }
        />
                <Route path="/budget" element={
                   <AuthRoute>
                  <Budget />
                  </AuthRoute>
                  } 
                  />
        <Route
          path="/dashboard"
          element={
            <AuthRoute>
              <Dashboard />
            </AuthRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <AuthRoute>
              <UserProfile />
            </AuthRoute>
          }
        /><Route
        path="/family"
        element={
          <AuthRoute>
            <FamilyManager />
          </AuthRoute>
        }
      />
      <Route
        path="/create-family"
        element={
          <AuthRoute>
            <CreateFamily />
          </AuthRoute>
        }
      />
           <Route
            path="/FamilyGoalForm"
             element={
             <AuthRoute>
                <FamilyGoals />
             </AuthRoute>} 
            />
        
        <Route path="/manage-subscriptions"
         element={
          <AuthRoute>
               <ManageSubscriptions />
          </AuthRoute>
        }
        />
    

      </Routes>
    </BrowserRouter>
  );
}

export default App;
