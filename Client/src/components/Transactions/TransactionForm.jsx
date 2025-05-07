


import React, { useState } from "react";
import { useFormik } from "formik";
import { useNavigate } from "react-router-dom";
import * as Yup from "yup";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  FaDollarSign,
  FaCalendarAlt,
  FaRegCommentDots,
  FaWallet,
  FaCreditCard,
  FaUsers,
} from "react-icons/fa";
import { listCategoriesAPI } from "../../services/category/categoryService";
import { listFamilyMembersAPI } from "../../services/familyService/familyService";
import { addTransactionAPI } from "../../services/transactions/transactionService";
import AlertMessage from "../Alert/AlertMessage";

const validationSchema = Yup.object({
  type: Yup.string()
    .required("Transaction type is required")
    .oneOf(["income", "expense"]),
  amount: Yup.number()
    .required("Amount is required")
    .positive("Amount must be positive"),
  category: Yup.string().required("Category is required"),
  date: Yup.date().required("Date is required"),
  description: Yup.string(),
  paymentMethod: Yup.string()
    .required("Payment method is required")
    .oneOf(["Cash", "UPI", "Credit Card", "Debit Card", "Bank Transfer", "Other"]),
  familyMember: Yup.string(),
});

const TransactionForm = () => {
  const navigate = useNavigate();
  const [splitBill, setSplitBill] = useState(false);
  const [selectedMembers, setSelectedMembers] = useState([]);
  const [warning, setWarning] = useState('');

  // Get user data from local storage
  const user = JSON.parse(localStorage.getItem("userInfo"));

  // Fetch categories
  const {
    data: categories,
    isError: isCategoryError,
    isLoading: isCategoryLoading,
    error: categoryError,
  } = useQuery({
    queryFn: listCategoriesAPI,
    queryKey: ["list-categories"],
  });

  // Fetch family members
  const {
    data: familyMembers,
    isError: isFamilyError,
    isLoading: isFamilyLoading,
    error: familyError,
  } = useQuery({
    queryFn: () => listFamilyMembersAPI(user.id),
    queryKey: ["list-family-members", user.id],
    enabled: !!user.id, // Prevents execution if userId is not available
  });

  // Mutation for adding a transaction
  const {
    mutateAsync,
    isPending,
    isError: isAddTranErr,
    error: transErr,
    isSuccess,
  } = useMutation({
    mutationFn: addTransactionAPI,
    mutationKey: ["add-transaction"],
  });

  const formik = useFormik({
    initialValues: {
      type: "",
      amount: "",
      category: "",
      date: "",
      description: "",
      paymentMethod: "",
      familyMember: "",
      splitBill: false,
      selectedMembers: [],
    },
    validationSchema,
    validateOnChange: false,
    validateOnBlur: false,
    
    onSubmit: async (values) => {
      try {
        console.log("Submitting values:", values);
        
        let transactions = [];

        if (splitBill && selectedMembers.length > 0) {
          const splitAmount = values.amount / selectedMembers.length; // Divide equally
          console.log("selected member:", selectedMembers);

          // Create separate transactions for each selected user
           transactions = selectedMembers.map((member) => ({
            ...values,
            user: member._id, // Assign each transaction to a family member
            splitUsers: selectedMembers.map((m) => m._id), // Store all selected user IDs
            splitExpense: true,
            amount: splitAmount, // Ensure amount is divided correctly
          }));
          console.log("Generated transactions:", transactions);
          // Send multiple transactions to API
          await Promise.all(transactions.map((tx) => mutateAsync(tx)));
        } else {
          // If no splitting, create a single transaction
          transactions = [
            {
              ...values,
              user: user.id,
              splitUsers: [],
              splitExpense: false,
              amount: values.amount,
            },
          ];
        }
    
        for (const tx of transactions) {
          const response = await mutateAsync(tx);
    
          // Check if response contains a warning
          if (response?.warning) {
            setWarning(response.warning);
          } else {
            setWarning(""); // Clear warning if no warning is returned
          }
        }
        console.log("Transactions successfully added");
      } catch (error) {
        console.error("Error adding transactions:", error);
      }
    }
    


    
    
  });
              const handleMemberSelection = (member) => {
                console.log("Member Clicked:", member);
              
                setSelectedMembers((prev) => {
                  let updatedMembers;
              
                  if (prev.some((m) => m._id === member._id)) {
                    updatedMembers = prev.filter((m) => m._id !== member._id);
                  } else {
                    updatedMembers = [...prev, member];
                  }
              
                  console.log("Updated Selected Members:", updatedMembers);
              
                  // Calculate split amount
                  const splitAmount = updatedMembers.length > 0 ? formik.values.amount / updatedMembers.length : 0;
                  console.log("Calculated Split Amount:", splitAmount);
              
                  // Update Formik's splitUsers field
                  formik.setFieldValue(
                    "splitUsers",
                    updatedMembers.map((m) => ({
                      userId: m._id,
                      amount: splitAmount,
                    }))
                  );
              
                  return updatedMembers;
                });
              };
  
            
            // Ensure amount updates when changed
            React.useEffect(() => {
              console.log("Selected Members Updated:", selectedMembers);
            
              if (selectedMembers.length > 0) {
                const splitAmount = formik.values.amount / selectedMembers.length;
                console.log("Amount:", formik.values.amount, "Split Amount:", splitAmount);
            
                formik.setFieldValue(
                  "splitUsers",
                  selectedMembers.map((m) => ({
                    userId: m._id,
                    amount: splitAmount,
                  }))
                );
              }
            }, [formik.values.amount, selectedMembers]);
            
  return (
    <form
    onSubmit={(e) => { 
      e.preventDefault(); 
      console.log("Form submit triggered"); 
      console.log("Formik Values Before Submit:", formik.values);
      console.log("Formik Errors Before Submit:", formik.errors);
      formik.handleSubmit(); 
    }}
    noValidate
    className="max-w-lg mx-auto my-10 bg-white p-6 rounded-lg shadow-lg space-y-6"
  >
  
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-gray-800">Transaction Details</h2>
        <p className="text-gray-600">Fill in the details below.</p>
      </div>


      {warning && (
        <AlertMessage
          type="warning"
          message={warning}
        />
      )}


      {isCategoryError && (
        <AlertMessage
          type="error"
          message={categoryError?.response?.data?.message || "Error fetching categories"}
        />
      )}
      {isFamilyError && (
        <AlertMessage
          type="error"
          message={familyError?.response?.data?.message || "Error fetching family members"}
        />
      )}
      {isSuccess && <AlertMessage type="success" message="Transaction added successfully" />}

      {/* Transaction Type */}
      <div className="space-y-2">
        <label htmlFor="type" className="flex gap-2 items-center text-gray-700 font-medium">
          <FaWallet className="text-blue-500" />
          <span>Type</span>
        </label>
        <select
          {...formik.getFieldProps("type")}
          id="type"
          className="block w-full p-2 mt-1 border border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-500"
        >
          <option value="">Select transaction type</option>
          <option value="income">Income</option>
          <option value="expense">Expense</option>
        </select>
      </div>

      {/* Amount */}
      <div className="flex flex-col space-y-1">
        <label htmlFor="amount" className="text-gray-700 font-medium">
          <FaDollarSign className="inline mr-2 text-blue-500" />
          Amount
        </label>
        <input
          type="number"
          {...formik.getFieldProps("amount")}
          id="amount"
          placeholder="Amount"
          className="w-full border border-gray-300 rounded-md py-2 px-3 focus:border-blue-500 focus:ring focus:ring-blue-500"
        />
      </div>

      {/* Category */}
      <div className="flex flex-col space-y-1">
        <label htmlFor="category" className="text-gray-700 font-medium">
          <FaRegCommentDots className="inline mr-2 text-blue-500" />
          Category
        </label>
        <select
          {...formik.getFieldProps("category")}
          id="category"
          className="w-full border border-gray-300 rounded-md py-2 px-3 focus:border-blue-500 focus:ring focus:ring-blue-500"
        >
          <option value="">Select a category</option>
          {categories?.map((category) => (
            <option key={category._id} value={category.name}>
              {category.name}
            </option>
          ))}
        </select>
      </div>

      {/* Payment Method */}
      <div className="flex flex-col space-y-1">
        <label htmlFor="paymentMethod" className="text-gray-700 font-medium">
          <FaCreditCard className="inline mr-2 text-blue-500" />
          Payment Method
        </label>
        <select
          {...formik.getFieldProps("paymentMethod")}
          id="paymentMethod"
          className="w-full border border-gray-300 rounded-md py-2 px-3 focus:border-blue-500 focus:ring focus:ring-blue-500"
        >
          <option value="">Select payment method</option>
          <option value="Cash">Cash</option>
          <option value="UPI">UPI</option>
          <option value="Credit Card">Credit Card</option>
          <option value="Debit Card">Debit Card</option>
          <option value="Bank Transfer">Bank Transfer</option>
          <option value="Other">Other</option>
        </select>
      </div>

      {/* Date */}
      <div className="flex flex-col space-y-1">
        <label htmlFor="date" className="text-gray-700 font-medium">
          <FaCalendarAlt className="inline mr-2 text-blue-500" />
          Date
        </label>
        <input
          type="date"
          {...formik.getFieldProps("date")}
          id="date"
          className="w-full border border-gray-300 rounded-md py-2 px-3 focus:border-blue-500 focus:ring focus:ring-blue-500"
        />
      </div>

      {/* Split Bill Checkbox */}
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="splitBill"
          checked={splitBill}
          onChange={() => setSplitBill(!splitBill)}
        />
        <label htmlFor="splitBill" className="text-gray-700 font-medium">
          Split this bill with family members?
        </label>
      </div>

      {/* Family Members List */}
      {splitBill && (
        <div>
          <label className="text-gray-700 font-medium">Select Family Members</label>
          {familyMembers?.members?.map((member) => (
            <div key={member._id} className="flex gap-2 items-center">
              <input type="checkbox" checked={selectedMembers.some((m) => m._id === member._id)} onChange={() => handleMemberSelection(member)} />
              <span>{member.username}</span>
            </div>
          ))}
        </div>
      )}
       {/* Description Field */}
      <div className="flex flex-col space-y-1">
        <label htmlFor="description" className="text-gray-700 font-medium">
           <FaRegCommentDots className="inline mr-2 text-blue-500" />
           Description (Optional)
         </label>
         <textarea
           {...formik.getFieldProps("description")}
           id="description"
           rows="3"
           className="w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50"
         ></textarea>
       </div> 
      <button type="submit" className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Submit Transaction
      </button>
    </form>
  );
};

export default TransactionForm;
