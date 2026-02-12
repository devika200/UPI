import React, { useEffect, useState, useMemo } from "react";
import axios from "axios";
import "./History.css";

const History = () => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: "timestamp", direction: "desc" });
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        setLoading(true);
        // Add a timeout for demonstration purposes
        const response = await axios.get("http://localhost:5000/api/history");
        setTransactions(response.data);
        setError(null);
      } catch (error) {
        console.error("Error fetching transactions:", error);
        setError("Failed to fetch transaction history. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    fetchTransactions();
  }, []);

  const handleSort = (key) => {
    let direction = "asc";
    if (sortConfig.key === key && sortConfig.direction === "asc") {
      direction = "desc";
    }
    setSortConfig({ key, direction });
  };

  const filteredTransactions = useMemo(() => {
    return transactions.filter((txn) => {
      return (
        txn.sender.toLowerCase().includes(searchTerm.toLowerCase()) ||
        txn.receiver.toLowerCase().includes(searchTerm.toLowerCase()) ||
        txn.amount.toString().includes(searchTerm)
      );
    });
  }, [transactions, searchTerm]);

  const sortedTransactions = useMemo(() => {
    const sortableTransactions = [...filteredTransactions];
    sortableTransactions.sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) {
        return sortConfig.direction === "asc" ? -1 : 1;
      }
      if (a[sortConfig.key] > b[sortConfig.key]) {
        return sortConfig.direction === "asc" ? 1 : -1;
      }
      return 0;
    });
    return sortableTransactions;
  }, [filteredTransactions, sortConfig]);

  // Calculate pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentTransactions = sortedTransactions.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(sortedTransactions.length / itemsPerPage);

  const formatDate = (dateString) => {
    const options = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit' 
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  const formatAmount = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  return (
    <div className="history-container">
      <div className="history-header">
        <h2>Transaction History</h2>
        <div className="search-container">
          <input
            type="text"
            placeholder="Search by sender, receiver, or amount..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
        </div>
      </div>

      {loading ? (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading transactions...</p>
        </div>
      ) : error ? (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>Try Again</button>
        </div>
      ) : (
        <>
          <div className="table-container">
            <table className="transactions-table">
              <thead>
                <tr>
                  <th onClick={() => handleSort("amount")} className="sortable-header">
                    Amount {sortConfig.key === "amount" && (sortConfig.direction === "asc" ? "↑" : "↓")}
                  </th>
                  <th onClick={() => handleSort("sender")} className="sortable-header">
                    Sender {sortConfig.key === "sender" && (sortConfig.direction === "asc" ? "↑" : "↓")}
                  </th>
                  <th onClick={() => handleSort("receiver")} className="sortable-header">
                    Receiver {sortConfig.key === "receiver" && (sortConfig.direction === "asc" ? "↑" : "↓")}
                  </th>
                  <th onClick={() => handleSort("timestamp")} className="sortable-header">
                    Timestamp {sortConfig.key === "timestamp" && (sortConfig.direction === "asc" ? "↑" : "↓")}
                  </th>
                  <th onClick={() => handleSort("is_fraud")} className="sortable-header">
                    Status {sortConfig.key === "is_fraud" && (sortConfig.direction === "asc" ? "↑" : "↓")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {currentTransactions.length > 0 ? (
                  currentTransactions.map((txn, index) => (
                    <tr 
                      key={index} 
                      className={txn.is_fraud ? "fraud-transaction" : "legitimate-transaction"}
                    >
                      <td className="amount-cell">{formatAmount(txn.amount)}</td>
                      <td>{txn.sender}</td>
                      <td>{txn.receiver}</td>
                      <td>{formatDate(txn.timestamp)}</td>
                      <td>
                        <span className={`status-badge ${txn.is_fraud ? "fraud" : "legitimate"}`}>
                          {txn.is_fraud ? "Fraud" : "Legitimate"}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="5" className="no-results">
                      No transactions found matching your search.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {sortedTransactions.length > itemsPerPage && (
            <div className="pagination">
              <button 
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="pagination-button"
              >
                Previous
              </button>
              <span className="page-info">
                Page {currentPage} of {totalPages}
              </span>
              <button 
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="pagination-button"
              >
                Next
              </button>
            </div>
          )}

          <div className="summary-section">
            <p><strong>Total Transactions:</strong> {sortedTransactions.length}</p>
            <p><strong>Fraud Transactions:</strong> {sortedTransactions.filter(txn => txn.is_fraud).length}</p>
            <p><strong>Legitimate Transactions:</strong> {sortedTransactions.filter(txn => !txn.is_fraud).length}</p>
          </div>
        </>
      )}
    </div>
  );
};

export default History;
