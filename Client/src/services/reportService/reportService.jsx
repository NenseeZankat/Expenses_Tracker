import axios from "axios";

export const generateTaxReport = async (userId, year) => {
    try {
        const response = await axios.get(`/api/reports/generate/${userId}/${year}`);
        return response.data;
    } catch (error) {
        console.error("Error generating tax report", error);
        throw error;
    }
};
