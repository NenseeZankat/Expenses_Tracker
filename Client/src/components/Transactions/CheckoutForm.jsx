import { PaymentElement } from "@stripe/react-stripe-js";
import { useEffect, useState } from "react";
import { useStripe, useElements } from "@stripe/react-stripe-js";
import { useNavigate } from "react-router-dom";
import { createPaymentIntentAPI } from "../../services/transactions/paymentService";
import { useFormik } from "formik";
import { useSelector } from "react-redux";
import * as Yup from "yup";

export default function CheckoutForm() {
  //const user = useSelector(state => state.auth.user);
  const user = JSON.parse(localStorage.getItem("userInfo"));
  const stripe = useStripe();
  const elements = useElements();
  const navigate = useNavigate();

  const [message, setMessage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [location, setLocation] = useState({ place: "", latitude: "", longitude: "" });

  const VITE_STRIPE_PUBLISHABLE_KEY = import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY;
  const VITE_GEOAPI_KEY = import.meta.env.VITE_GEOAPI_KEY;
  const MAPBOX_API_KEY = import.meta.env.VITE_MAPBOX_API_KEY;
  

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          setLocation((prev) => ({ ...prev, latitude, longitude }));
          await fetchPlaceFromCoordinates(latitude, longitude);
        },
        (error) => {
          console.error("Error getting geolocation:", error);
          setMessage("Failed to get current location");
        }
      );
    } else {
      setMessage("Geolocation is not supported by your browser");
    }
  }, []);

  const fetchPlaceFromCoordinates = async (lat, lon) => {
    try {
      const response = await fetch(
        `https://api.mapbox.com/geocoding/v5/mapbox.places/${lon},${lat}.json?access_token=${MAPBOX_API_KEY}`
      );
      const data = await response.json();
      if (data.features?.length > 0) {
        setLocation((prev) => ({
          ...prev,
          place: data.features[0].place_name,
        }));
      }
    } catch (error) {
      console.error("Error fetching place name:", error);
      setMessage("Failed to fetch place name");
    }
  };

  const fetchLocationFromMapbox = async (place) => {
    try {
      const response = await fetch(
        `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(place)}.json?access_token=${MAPBOX_API_KEY}`
      );
      const data = await response.json();
      if (data.features?.length > 0) {
        const feature = data.features[0];
        setLocation({
          place: feature.place_name,
          latitude: feature.center[1].toFixed(6),
          longitude: feature.center[0].toFixed(6),
        });
      } else {
        setMessage("Location not found");
      }
    } catch (error) {
      console.error("Error fetching location from Mapbox:", error);
      setMessage("Failed to fetch location");
    }
  };

  const handleLocationInput = () => {
    const place = formik.values.placeInput.trim();
    if (place) {
      fetchLocationFromMapbox(place);
    } else {
      setMessage("Please enter a location");
    }
  };

  const formik = useFormik({
    initialValues: {
      amount: "",
      category: "",
      type: "",
      placeInput: "",
    },
    validationSchema: Yup.object({
      amount: Yup.number().required("Amount is required").positive("Amount must be positive"),
      category: Yup.string().required("Category is required"),
      type: Yup.string().required("Transaction type is required"),
      placeInput: Yup.string().nullable(),
    }),
    onSubmit: async (values) => {
      if (!stripe || !elements) return;

      if (!user || !user?.id) {
        console.log(user);
        setMessage("User not found. Please login."+user.id);
        return;
      }

      setIsProcessing(true);

      const transactionData = {
        amount: values.amount,
        category: values.category,
        type: values.type,
        placeName: `${values.placeInput || "Unknown place"} in ${location.place || "Unknown location"}`,
        user: user.id,
        description:`Payment of $${values.amount} made via card`,
        //latitude: location.latitude,
        //longitude: location.longitude,
        paymentMethod:"Credit Card",
      };

      try {
        const res = await createPaymentIntentAPI(transactionData);
        localStorage.setItem("transactionData", JSON.stringify(transactionData));
        const clientSecret = res?.clientSecret;
      
        if (!clientSecret) throw new Error("No client secret received");
      
        await elements.submit();

        const result = await stripe.confirmPayment({
          elements,
          clientSecret, // ✅ Pass the client secret here
          confirmParams: {
            return_url: window.location.origin + "/completion",
          },
        });
      
        if (result.error) {
          setMessage(result.error.message);
        } else {
          setMessage("Payment completed!");
          
          navigate("/completion");
        }
      } catch (error) {
        console.error("Error processing transaction:", error);
        setMessage(error?.response?.data?.message || error.message || "Payment failed.");
      } finally {
        setIsProcessing(false);
      }
      
    },
  });

  return (
    <form
      onSubmit={formik.handleSubmit}
      className="max-w-lg mx-auto my-10 bg-white p-6 rounded-lg shadow-lg space-y-6"
    >
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-gray-800">Checkout Form</h2>
        <p className="text-gray-600">Complete your transaction below.</p>
      </div>

      <input
        type="number"
        placeholder="Amount"
        {...formik.getFieldProps("amount")}
        className="w-full p-2 border border-gray-300 rounded-md"
      />
      {formik.touched.amount && formik.errors.amount && (
        <div className="text-red-500">{formik.errors.amount}</div>
      )}

      <select
        {...formik.getFieldProps("category")}
        className="w-full p-2 border border-gray-300 rounded-md"
      >
        <option value="">Select Category</option>
        <option value="shopping">Shopping</option>
        <option value="food">Food</option>
        <option value="entertainment">Entertainment</option>
        <option value="other">Other</option>
      </select>
      {formik.touched.category && formik.errors.category && (
        <div className="text-red-500">{formik.errors.category}</div>
      )}

      <select
        {...formik.getFieldProps("type")}
        className="w-full p-2 border border-gray-300 rounded-md"
      >
        <option value="">Select Type</option>
        <option value="income">Income</option>
        <option value="expense">Expense</option>
      </select>
      {formik.touched.type && formik.errors.type && (
        <div className="text-red-500">{formik.errors.type}</div>
      )}

      <div className="flex space-x-2">
        <input
          type="text"
          placeholder="Enter location"
          {...formik.getFieldProps("placeInput")}
          className="w-full p-2 border border-gray-300 rounded-md"
        />
        <button
          type="button"
          onClick={handleLocationInput}
          className="bg-blue-500 text-white p-2 rounded-md hover:bg-blue-700"
        >
          Get Location
        </button>
      </div>

      <PaymentElement id="payment-element" />

      <div className="mt-4">
        <strong>Selected Location:</strong> {location.place || "Not selected"}
      </div>
      <div>
        <strong>Latitude:</strong> {location.latitude}
      </div>
      <div>
        <strong>Longitude:</strong> {location.longitude}
      </div>

      <button
        disabled={isProcessing || !stripe || !elements || !formik.isValid}
        className={`w-full py-2 px-4 rounded font-bold ${
          formik.isValid
            ? "bg-blue-500 hover:bg-blue-700 text-white"
            : "bg-gray-400 text-gray-200"
        }`}
      >
        {isProcessing ? "Processing..." : "Pay now"}
      </button>

      {message && <div className="text-red-500">{message}</div>}
    </form>
  );
}
