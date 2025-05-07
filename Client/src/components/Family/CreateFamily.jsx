import { useState } from "react";
import { createFamily } from "../../services/familyService/familyService";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";

const CreateFamily = () => {
    const user = useSelector(state => state.auth.user);
    const [familyName, setFamilyName] = useState("");
    const [error, setError] = useState(""); // State for error message
    const navigate = useNavigate();

    // const handleCreate = async () => {
    //     await createFamily(familyName, user.id);
    //     navigate("/");
    // };

    const handleCreate = async () => {
        setError(""); // Reset error before request
        try {
            await createFamily(familyName, user.id);
            navigate("/");
        } catch (err) {
            // Show error message from backend
            setError(err.response?.data?.message || "Failed to create family.");
        }
    };

    return (
        <div className="p-5">
           <h2 className="text-xl font-bold">Create a Family</h2>
           {error && <p className="text-red-500">{error}</p>} {/* Show error message */}
            <input
                type="text"
                placeholder="Family Name"
                className="border p-2 rounded"
                value={familyName}
                onChange={(e) => setFamilyName(e.target.value)}
            />
            <button onClick={handleCreate} className="px-4 py-2 bg-green-500 text-white rounded ml-2">Create</button>
        </div>
    );
};

export default CreateFamily;
