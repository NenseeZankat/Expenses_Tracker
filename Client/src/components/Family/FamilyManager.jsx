import { useEffect, useState } from "react";
import { getFamilyDetails, addMember, removeMember, assignRole } from "../../services/familyService/familyService";
import { useSelector, useDispatch } from "react-redux";
import { updateUser } from "../../redux/slice/authSlice";
import axios from "axios";

const API_URL = "https://backend-johh.onrender.com";

const FamilyManager = () => {
    const user = useSelector(state => state.auth.user);
    const dispatch = useDispatch();

    const [family, setFamily] = useState({ members: [] });
    const [newMemberEmail, setNewMemberEmail] = useState(""); // Now email input
    const [transactions, setTransactions] = useState({});
    const [loadingFamily, setLoadingFamily] = useState(true);
    const [loadingTransactions, setLoadingTransactions] = useState({});

    useEffect(() => {
        if (user?.familyId) {
            fetchFamilyData();
        }
    }, [user.familyId]);

    const fetchFamilyData = async () => {
        try {
            setLoadingFamily(true);
            const data = await getFamilyDetails(user.familyId);
            if (data && data.family) {
                setFamily(data.family);
            } else {
                console.warn("Invalid family data structure:", data);
            }
        } catch (error) {
            console.error("Error fetching family data:", error);
        } finally {
            setLoadingFamily(false);
        }
    };

    useEffect(() => {           
        const fetchUserData = async () => {
            try {
                const userResponse = await axios.get(`${API_URL}/api/v1/users/profile`, {
                    headers: {
                        Authorization: `Bearer ${localStorage.getItem("token")}`
                    }
                });

                const updatedUser = userResponse.data;
                let storedUser = JSON.parse(localStorage.getItem("userInfo")) || {};
                storedUser.familyId = updatedUser.familyId;
                localStorage.setItem("userInfo", JSON.stringify(storedUser));
                dispatch(updateUser(storedUser));
            } catch (error) {
                console.error("Error fetching user data:", error);
            }
        };

        if (user?.id) {
            fetchUserData();
        }
    }, [user?.id, dispatch]);

    const fetchTransactions = async (memberId) => {
        setLoadingTransactions((prev) => ({ ...prev, [memberId]: true }));
        try {
            const res = await axios.get(`${API_URL}/api/v1/transactions/user/${memberId}`);
            setTransactions((prev) => ({ ...prev, [memberId]: res.data }));
        } catch (error) {
            console.error("Error fetching transactions:", error);
        } finally {
            setLoadingTransactions((prev) => ({ ...prev, [memberId]: false }));
        }
    };

    const handleAddMember = async () => {
        try {
            if (!newMemberEmail) {
                alert("Please enter an email address.");
                return;
            }

            // Find user by email
            console.log(newMemberEmail);
            const res = await axios.get(`${API_URL}/api/v1/users/by-email/${newMemberEmail}`);

            const foundUser = res.data;
            console.log(foundUser);
            if (!foundUser || !foundUser.id) {
                alert("User not found with this email.");
                return;
            }

            await addMember(family._id, user.id, foundUser.id);
            setNewMemberEmail("");
            fetchFamilyData();
        } catch (error) {
            console.error("Error adding member:", error);
            alert("Failed to add member. Make sure email is correct.");
        }
    };

    const handleRemoveMember = async (id) => {
        await removeMember(family._id, user.id, id);
        fetchFamilyData();
    };

    const handleAssignRole = async (id, role) => {
        await assignRole(family._id, user.id, id, role);
        fetchFamilyData();
    };

    if (loadingFamily) return <p>Loading family data...</p>;

    return (
        <div className="p-5">
            <h2 className="text-2xl font-bold mb-2 text-blue-700">
                {family.familyName ? `${family.familyName} Family` : "Your Family"}
            </h2>
            <h2 className="text-xl font-bold">Manage Family</h2>

            <ul className="mt-4">
                {Array.isArray(family.members) && family.members.length > 0 ? (
                    family.members.map(member => (
                        <li key={member._id} className="p-2 border-b">
                            <div className="flex justify-between">
                                <span>{member.username} ({member.role})</span>
                                {user.id === family.admin && (
                                    <div>
                                        <button onClick={() => handleAssignRole(member._id, "admin")} className="mx-1 px-3 py-1 bg-blue-500 text-white rounded">Make Admin</button>
                                        <button onClick={() => handleAssignRole(member._id, "member")} className="mx-1 px-3 py-1 bg-yellow-500 text-white rounded">Make Member</button>
                                        <button onClick={() => handleRemoveMember(member._id)} className="mx-1 px-3 py-1 bg-red-500 text-white rounded">Remove</button>
                                    </div>
                                )}
                            </div>

                            <button 
                                onClick={() => fetchTransactions(member._id)} 
                                className="mt-2 px-3 py-1 bg-gray-500 text-white rounded"
                                disabled={loadingTransactions[member._id]}
                            >
                                {loadingTransactions[member._id] ? "Loading..." : "Show Transactions"}
                            </button>

                            {transactions[member._id] && (
                                <div className="mt-2 p-3 border rounded-lg bg-gray-100">
                                    <h3 className="font-semibold mb-2">Transactions</h3>
                                    <ul>
                                        {transactions[member._id].length > 0 ? (
                                            transactions[member._id].map((txn, index) => (
                                                <li 
                                                    key={index} 
                                                    className={`flex justify-between items-center p-2 mb-1 rounded-lg ${
                                                        txn.type === "expense" ? "bg-red-100 text-red-600" : "bg-green-100 text-green-600"
                                                    }`}
                                                >
                                                    <span>{new Date(txn.date).toLocaleDateString()}</span>
                                                    <span className="font-bold">${txn.amount}</span>
                                                    <span className="text-xs font-semibold uppercase">{txn.type}</span>
                                                </li>
                                            ))
                                        ) : (
                                            <p className="text-gray-500 text-sm">No transactions found.</p>
                                        )}
                                    </ul>
                                </div>
                            )}
                        </li>
                    ))
                ) : (
                    <p>No family members found.</p>
                )}
            </ul>

            {user.id === family.admin && (
                <div className="mt-4">
                    <input
                        type="email"
                        placeholder="Enter member's email"
                        className="border p-2 rounded"
                        value={newMemberEmail}
                        onChange={(e) => setNewMemberEmail(e.target.value)}
                    />
                    <button onClick={handleAddMember} className="px-4 py-2 bg-green-500 text-white rounded ml-2">Add Member</button>
                </div>
            )}
        </div>
    );
};

export default FamilyManager;
