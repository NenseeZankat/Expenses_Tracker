import React, { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { FaTrash, FaEdit } from "react-icons/fa";
import { ChevronDownIcon } from "@heroicons/react/24/solid";

import {
  listTransactionsAPI,
  deleteTransactionAPI,
  updateTransactionAPI,
} from "../../services/transactions/transactionService";
import { listCategoriesAPI } from "../../services/category/categoryService";

const TransactionList = () => {
  // Filtering state
  const [filters, setFilters] = useState({
    startDate: "",
    endDate: "",
    type: "",
    category: "",
  });

  // Editing state
  const [editingTransactionId, setEditingTransactionId] = useState(null);
  const [editData, setEditData] = useState({
    type: "",
    category: "",
    description: "",
  });

  // Handle filter change
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters((prev) => ({ ...prev, [name]: value }));
  };

  // Fetch categories
  const { data: categoriesData, isLoading: categoryLoading } = useQuery({
    queryFn: listCategoriesAPI,
    queryKey: ["list-categories"],
  });

  // Fetch transactions
  const {
    data: transactions,
    isLoading: transactionsLoading,
    refetch,
  } = useQuery({
    queryFn: () => listTransactionsAPI(filters),
    queryKey: ["list-transactions", filters],
  });

  // Delete mutation
  const deleteTransactionMutation = useMutation({
    mutationFn: (transactionId) => deleteTransactionAPI(transactionId),
    onSuccess: () => refetch(),
    onError: (error) => {
      console.error("Delete failed:", error);
      alert(error.response?.data?.message || "Failed to delete transaction");
    },
  });

  // Update mutation
  const updateTransactionMutation = useMutation({
    mutationFn: ({ id, data }) => updateTransactionAPI(id, data),
    onSuccess: () => {
      setEditingTransactionId(null);
      refetch();
    },
    onError: (error) => {
      console.error("Update failed:", error);
      alert(error.response?.data?.message || "Failed to update transaction");
    },
  });

  // Handle delete
  const handleDelete = (transactionId) => {
    deleteTransactionMutation.mutate(transactionId);
  };

  // Handle edit click
  const handleEditClick = (transaction) => {
    setEditingTransactionId(transaction._id);
    setEditData({
      type: transaction.type,
      category: transaction.category || "",
      description: transaction.description || "",
    });
  };

  // Handle edit form change
  const handleEditChange = (e) => {
    const { name, value } = e.target;
    setEditData((prev) => ({ ...prev, [name]: value }));
  };

  // Handle update submit
  const handleUpdateSubmit = (transactionId) => {
    const updatedData = {
      type: editData.type,
      description: editData.description,
      category: editData.category,
    };
    updateTransactionMutation.mutate({ id: transactionId, data: updatedData });
  };

  return (
    <div className="my-4 p-4 shadow-lg rounded-lg bg-white">
      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <input
          type="date"
          name="startDate"
          value={filters.startDate}
          onChange={handleFilterChange}
          className="p-2 rounded-lg border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50"
        />
        <input
          type="date"
          name="endDate"
          value={filters.endDate}
          onChange={handleFilterChange}
          className="p-2 rounded-lg border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50"
        />
        <div className="relative">
          <select
            name="type"
            value={filters.type}
            onChange={handleFilterChange}
            className="w-full p-2 rounded-lg border-gray-300 appearance-none focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50"
          >
            <option value="">All Types</option>
            <option value="income">Income</option>
            <option value="expense">Expense</option>
          </select>
          <ChevronDownIcon className="w-5 h-5 absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-500" />
        </div>
        <div className="relative">
          <select
            name="category"
            value={filters.category}
            onChange={handleFilterChange}
            className="w-full p-2 rounded-lg border-gray-300 appearance-none focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50"
          >
            <option value="">All Categories</option>
            {categoriesData?.map((category) => (
              <option key={category._id} value={category.name}>
                {category.name}
              </option>
            ))}
          </select>
          <ChevronDownIcon className="w-5 h-5 absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-500" />
        </div>
      </div>

      {/* Transactions */}
      <div className="my-4">
        <h3 className="text-xl font-semibold mb-4">Transactions</h3>
        {transactionsLoading ? (
          <div>Loading...</div>
        ) : (
          <ul className="space-y-4">
            {transactions?.map((transaction) => (
              <li
                key={transaction._id}
                className="bg-white p-4 rounded-lg shadow flex flex-col gap-2 border"
              >
                {editingTransactionId === transaction._id ? (
                  <>
                    {/* Edit Form */}
                    <div className="flex flex-col gap-2">
                      <select
                        name="type"
                        value={editData.type}
                        onChange={handleEditChange}
                        className="p-2 rounded-lg border-gray-300"
                      >
                        <option value="income">Income</option>
                        <option value="expense">Expense</option>
                      </select>
                      <select
                        name="category"
                        value={editData.category}
                        onChange={handleEditChange}
                        className="p-2 rounded-lg border-gray-300"
                      >
                        {categoriesData?.map((cat) => (
                          <option key={cat._id} value={cat.name}>
                            {cat.name}
                          </option>
                        ))}
                      </select>
                      <input
                        type="text"
                        name="description"
                        value={editData.description}
                        onChange={handleEditChange}
                        placeholder="Description"
                        className="p-2 rounded-lg border-gray-300"
                      />
                    </div>

                    {/* Save / Cancel Buttons */}
                    <div className="flex space-x-3 mt-2">
                      <button
                        onClick={() => handleUpdateSubmit(transaction._id)}
                        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                      >
                        Save
                      </button>
                      <button
                        onClick={() => setEditingTransactionId(null)}
                        className="px-4 py-2 bg-gray-300 text-black rounded-lg hover:bg-gray-400"
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    {/* Display */}
                    <div className="flex flex-col gap-1">
                      <span className="text-gray-700 font-semibold">
                        {new Date(transaction.date).toLocaleDateString()}
                      </span>
                      <span
                        className={`px-2 py-1 inline-block text-xs rounded-full ${
                          transaction.type === "income"
                            ? "bg-green-100 text-green-800"
                            : "bg-red-100 text-red-800"
                        }`}
                      >
                        {transaction.type.charAt(0).toUpperCase() +
                          transaction.type.slice(1)}
                      </span>
                      <span className="text-gray-800">
                        {transaction.category?.name} - $
                        {transaction.amount.toLocaleString()}
                      </span>
                      <span className="text-sm italic text-gray-600">
                        {transaction.description}
                      </span>
                      <span className="text-sm text-gray-600">
                        Split Expense: {transaction.splitExpense ? "Yes" : "No"}
                      </span>
                      <span className="text-sm text-gray-600">
                        Split Users:{" "}
                        {transaction.splitUsers?.length
                          ? transaction.splitUsers.join(", ")
                          : "None"}
                      </span>
                      <span className="text-sm text-gray-600">
                        User: {transaction.user?.name || "N/A"}
                      </span>
                    </div>

                    {/* Actions */}
                    <div className="flex space-x-3 mt-2">
                      <button
                        onClick={() => handleEditClick(transaction)}
                        className="text-blue-500 hover:text-blue-700"
                      >
                        <FaEdit />
                      </button>
                      <button
                        onClick={() => handleDelete(transaction._id)}
                        className="text-red-500 hover:text-red-700"
                      >
                        <FaTrash />
                      </button>
                    </div>
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default TransactionList;
